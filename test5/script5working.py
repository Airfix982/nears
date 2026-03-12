import wave
import numpy as np
import hashlib
import random
import math
from typing import Dict, List, Tuple
from scipy.interpolate import CubicSpline
import base64
import json
from scipy.signal import find_peaks
import script.utils as me


def read_wav_mono(path):
    with wave.open(path, "rb") as f:
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        n_frames = f.getnframes()
        comptype = f.getcomptype()
        raw = f.readframes(n_frames)
    
    if(comptype!="NONE"):
        raise ValueError("not compressed pcm wav only")
    if(n_channels != 1):
        raise ValueError("mono only")

    bits = 8 * sampwidth


    if sampwidth == 1:
        # 8-bit PCM в WAV обычно unsigned
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        min_val, max_val = 0, 255
        dtype_name = "uint8"
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.int64)
        min_val, max_val = -32768, 32767
        dtype_name = "int16"
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        min_val, max_val = -(1 << 31), (1 << 31) - 1
        dtype_name = "int32"

    meta = {
        "sample_rate": framerate,
        "sample_width_bytes": sampwidth,
        "bits_per_sample": bits,
        "min_value": min_val,
        "max_value": max_val,
        "dtype_name": dtype_name,
    }
    return arr, meta


def write_wav_mono(path, samples, meta):
    samples = clip_to_pcm_range(samples, meta)

    sampwidth = meta["sample_width_bytes"]
    framerate = meta["sample_rate"]


    if sampwidth == 1:
        raw = samples.astype(np.uint8).tobytes()
    elif sampwidth == 2:
        raw = samples.astype("<i2").tobytes()
    elif sampwidth == 4:
        raw = samples.astype("<i4").tobytes()

    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.writeframes(raw)


def clip_to_pcm_range(samples, meta):
    minimum = meta["min_value"]
    maximum = meta["max_value"]
    return np.clip(np.rint(samples), minimum, maximum).astype(np.int64)


def samples_to_bytes(samples, meta):
    samples = clip_to_pcm_range(samples, meta)
    sw = meta["sample_width_bytes"]
    if sw == 1:
        return samples.astype(np.uint8).tobytes()
    if sw == 2:
        return samples.astype("<i2").tobytes()
    if sw == 4:
        return samples.astype("<i4").tobytes()



# hash+ permutaion

def compute_chunk_hash(samples, meta):
    raw = samples_to_bytes(samples, meta)
    return hashlib.sha3_512(raw).digest()


def xor_two_byte_slices(h, start_a, start_b, size=8):
    a = int.from_bytes(h[start_a:start_a+size], "big")
    b = int.from_bytes(h[start_b:start_b+size], "big")
    return a ^ b


def derive_2dclm_params(h):
    if len(h) != 64:
        raise ValueError("SHA3-512 digest должен быть 64 байта")

    denom = float(2**64)
    x0 = xor_two_byte_slices(h, 0, 8) / denom
    y0 = xor_two_byte_slices(h, 16, 24) / denom
    r1 = xor_two_byte_slices(h, 32, 40) / denom + 2.9
    r2 = xor_two_byte_slices(h, 48, 56) / denom + 2.9

    eps = 1e-15
    x0 = min(max(x0, eps), 1-eps)
    y0 = min(max(y0, eps), 1-eps)
    return x0, y0, r1, r2


def build_permutation(length, h):
    seed =int.from_bytes(h[:16], "big")
    rng = random.Random(seed)
    idx = list(range(length))
    rng.shuffle(idx)
    return np.array(idx, dtype=np.int64)


def invert_permutation(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(permutation), dtype=permutation.dtype)
    return inv


def apply_scrambling(samples, permutation):
    return samples[permutation]


def apply_inverse_scrambling(samples, permutation):
    inv = invert_permutation(permutation)
    return samples[inv]



# 2dclm + keystream

def generate_2dclm(length, x0, y0, r1, r2, burn_in=1000):
    total = length + burn_in
    x = np.empty(total, dtype=np.float64)
    y = np.empty(total, dtype=np.float64)

    x[0] = x0
    y[0] = y0

    for i in range(total-1):
        xn = x[i]
        yn = y[i]
        x_next = r1 * math.cos(xn * (1.0 - xn * xn))
        y_next = r2 * math.cos(yn * (1.0 - xn))
        # берем дробную часть в [0, 1)
        x[i+1] = abs(x_next) % 1.0
        y[i+1] = abs(y_next) % 1.0

    return x[burn_in:], y[burn_in:]


def build_keystream_words(x, y, bits_per_sample):
    if(len(x) != len(y)):
        raise ValueError("x y must be the same length")

    scale = 10**16
    xi = np.floor(x * scale).astype(np.uint64)
    yi = np.floor(y * scale).astype(np.uint64)
    mask = (1 << bits_per_sample) - 1

    k = ((xi ^ yi) & np.uint64(mask)).astype(np.uint64)
    return k


# matrix n padding

def pad_to_multiple(samples, multiple):
    pad_len = (-len(samples)) % multiple
    if(pad_len==0):
        return samples.copy(), 0

    padded = np.pad(samples, (0, pad_len), mode="reflect")
    return padded, pad_len


def reshape_to_matrix(samples: np.ndarray, row_len):
    if(len(samples) % row_len != 0):
        raise ValueError("Длина должна делиться на row_len")
    rows = len(samples) // row_len
    return samples.reshape(rows, row_len)


def flatten_matrix(mat):
    return mat.reshape(-1)

# EMD

def emd_decompose(signal, max_imfs=10, max_siftings=10, residue_std_threshold=1e-6):
    s = signal.astype(np.float64).copy()
    imfs: List[np.ndarray] = []

    for _ in range(max_imfs):
        if not has_enough_extrema(s):
            break

        h = s.copy()

        for _ in range(max_siftings):
            upper, lower = build_envelopes(h)
            mean_env = (upper + lower) / 2.0
            h1 = h - mean_env

            if is_imf(h1):
                h = h1
                break
            h=h1

        imfs.append(h.copy())
        s = s - h

        if np.std(s) < residue_std_threshold:
            break
    residue = s
    return imfs, residue



def has_enough_extrema(x):
    maxima, _ = find_peaks(x)
    minima, _ = find_peaks(-x)
    return len(maxima) + len(minima) >= 4


def build_envelopes(x):
    n = len(x)
    t = np.arange(n, dtype=np.float64)

    maxima, _ = find_peaks(x)
    minima, _ = find_peaks(-x)

    maxima = np.unique(np.concatenate(([0], maxima, [n-1])))
    minima = np.unique(np.concatenate(([0], minima, [n-1])))

    max_vals = x[maxima]
    min_vals = x[minima]

    upper = spline_or_linear(maxima.astype(np.float64), max_vals, t)
    lower = spline_or_linear(minima.astype(np.float64), min_vals, t)
    return upper, lower


def spline_or_linear(xs, ys, t):
    if len(xs) < 4:
        return np.interp(t, xs, ys)
    try:
        cs = CubicSpline(xs, ys, bc_type="natural")
        return cs(t)
    except Exception:
        return np.interp(t, xs, ys)


def is_imf(x):
    zero_crossings = np.sum(x[:-1] * x[1:] < 0)
    maxima, _ = find_peaks(x)
    minima, _ = find_peaks(-x)
    extrema = len(maxima) + len(minima)

    cond1 = abs(extrema - zero_crossings) <= 1

    upper, lower = build_envelopes(x)
    mean_env = (upper + lower) / 2.0
    cond2 = np.mean(np.abs(mean_env)) < max(1e-6, 0.01 * np.mean(np.abs(x)) + 1e-9)
    return cond1 and cond2


def reconstruct_from_imfs_and_residue(imfs, residue):
    if not imfs:
        return residue.copy()
    acc = np.sum(np.vstack(imfs), axis=0)
    return acc + residue


# Residue Encryption

def signed_to_unsigned_words(values, bits):
    mask = (1 << bits) - 1
    return (values.astype(np.int64) & mask).astype(np.uint64)

def unsigned_to_signed_words(values, bits):
    sign_bit = 1 << (bits - 1)
    full = 1 << bits
    v = values.astype(np.int64)
    return np.where(v >= sign_bit, v - full, v).astype(np.int64)

def float_residue_to_pcm_int(residue, meta):
    return clip_to_pcm_range(np.rint(residue), meta)


def encrypt_residue(residue, ks_row, meta, mode="xor"):
    bits = meta["bits_per_sample"]
    r_int = float_residue_to_pcm_int(residue, meta)

    if mode == "xor":
        r_u = signed_to_unsigned_words(r_int, bits)
        out_u = np.bitwise_xor(r_u, ks_row.astype(np.uint64))
        out_s = unsigned_to_signed_words(out_u, bits)
        return out_s.astype(np.float64)

    if mode == "modadd":
        mod = 1 << bits
        r_u = signed_to_unsigned_words(r_int, bits)
        out_u = (r_u + ks_row.astype(np.uint64)) % mod
        out_s = unsigned_to_signed_words(out_u, bits)
        return out_s.astype(np.float64)

    raise ValueError(f"Unknown mode: {mode}")


def decrypt_residue(residue_enc, ks_row, meta, mode="xor"):
    bits = meta["bits_per_sample"]
    r_int = float_residue_to_pcm_int(residue_enc, meta)

    if mode == "xor":
        r_u = signed_to_unsigned_words(r_int, bits)
        out_u = np.bitwise_xor(r_u, ks_row.astype(np.uint64))
        out_s = unsigned_to_signed_words(out_u, bits)
        return out_s.astype(np.float64)

    if mode == "modadd":
        mod = 1 << bits
        r_u = signed_to_unsigned_words(r_int, bits)
        out_u = (r_u - ks_row.astype(np.uint64)) % mod
        out_s = unsigned_to_signed_words(out_u, bits)
        return out_s.astype(np.float64)

    raise ValueError("unknown mod")

#per-row / per-chunk crypto
def encrypt_matrix_rows(mat, ks_mat, meta, itr, mode, emd_max_imfs, emd_max_siftings):
    out = mat.astype(np.float64).copy()

    for _ in range(itr):
        new_mat = np.empty_like(out)
        for i in range(out.shape[0]):
            sig = out[i, :]
            imfs, residue = emd_decompose(sig, max_imfs=emd_max_imfs, max_siftings=emd_max_siftings)
            residue_enc = encrypt_residue(residue, ks_mat[i, :], meta, mode=mode)
            sig_enc = reconstruct_from_imfs_and_residue(imfs, residue_enc)
            new_mat[i, :] = sig_enc
        out = new_mat
    return out


def decrypt_matrix_rows(mat, ks_mat, meta, itr, mode, emd_max_imfs, emd_max_siftings):
    out = mat.astype(np.float64).copy()

    for _ in range(itr):
        new_mat = np.empty_like(out)
        for i in range(out.shape[0]):
            sig = out[i, :]
            imfs, residue_enc = emd_decompose(sig, max_imfs=emd_max_imfs, max_siftings=emd_max_siftings)
            residue = decrypt_residue(residue_enc, ks_mat[i, :], meta, mode=mode)
            sig_dec = reconstruct_from_imfs_and_residue(imfs, residue)
            new_mat[i, :] = sig_dec
        out = new_mat
    return out


def encrypt_chunk(chunk_samples, meta, row_len, itr, mode, emd_max_imfs, emd_max_siftings):
    h = compute_chunk_hash(chunk_samples, meta)
    x0, y0, r1, r2 = derive_2dclm_params(h)

    perm = build_permutation(len(chunk_samples), h)
    scrambled = apply_scrambling(chunk_samples, perm)

    padded, pad_len = pad_to_multiple(scrambled, row_len)
    mat = reshape_to_matrix(padded.astype(np.float64), row_len)

    x, y = generate_2dclm(len(padded), x0, y0, r1, r2)
    ks = build_keystream_words(x, y, meta["bits_per_sample"])
    ks_mat = reshape_to_matrix(ks, row_len)

    enc_mat = encrypt_matrix_rows(mat=mat, ks_mat=ks_mat, meta=meta, itr=itr, mode=mode, emd_max_imfs=emd_max_imfs, emd_max_siftings=emd_max_siftings)
    enc_flat = flatten_matrix(enc_mat)
    if pad_len:
        enc_flat = enc_flat[:-pad_len]

    enc_int = clip_to_pcm_range(enc_flat, meta)
    return enc_int, h, pad_len

def decrypt_chunk(enc_chunk_samples, h, meta, row_len, itr, mode, pad_len, emd_max_imfs, emd_max_siftings):
    x0, y0, r1, r2 = derive_2dclm_params(h)

    padded, extra_pad = pad_to_multiple(enc_chunk_samples, row_len)
    if extra_pad != pad_len:
        pass

    mat = reshape_to_matrix(padded.astype(np.float64), row_len)

    x, y = generate_2dclm(len(padded), x0, y0, r1, r2)
    ks = build_keystream_words(x, y, meta["bits_per_sample"])
    ks_mat = reshape_to_matrix(ks, row_len)

    dec_mat = decrypt_matrix_rows(mat=mat, ks_mat=ks_mat, meta=meta, itr=itr, mode=mode, emd_max_imfs=emd_max_imfs, emd_max_siftings=emd_max_siftings)

    dec_flat = flatten_matrix(dec_mat)
    if pad_len:
        dec_flat = dec_flat[:-pad_len]

    dec_int = clip_to_pcm_range(dec_flat, meta)

    perm = build_permutation(len(dec_int), h)
    restored = apply_inverse_scrambling(dec_int, perm)
    return clip_to_pcm_range(restored, meta)



# sidecar with h_i

def protect_hashes(hashes: List[bytes], secret: str):
    secret_bytes = secret.encode("utf-8")
    out = []

    for i, h in enumerate(hashes):
        ks = derive_sidecar_keystream(secret_bytes, i, len(h))
        enc = bytes(a ^ b for a, b in zip(h, ks))
        out.append(base64.b64encode(enc).decode("ascii"))
    return out

def unprotect_hashes(encoded_hashes: List[str], secret: str):
    secret_bytes = secret.encode("utf-8")
    out = []

    for i, s in enumerate(encoded_hashes):
        enc = base64.b64decode(s.encode("ascii"))
        ks = derive_sidecar_keystream(secret_bytes, i, len(enc))
        h = bytes(a ^ b for a, b in zip(enc, ks))
        out.append(h)
    return out

def derive_sidecar_keystream(secret, counter, length):
    buf = bytearray()
    block_no = 0
    while(len(buf) < length):
        m = hashlib.sha3_512()
        m.update(secret)
        m.update(counter.to_bytes(8, "big"))
        m.update(block_no.to_bytes(8, "big"))
        buf.extend(m.digest())
        block_no += 1
    return bytes(buf[:length])

def save_hash_sidecar(path, hashes, pad_lens, secret, config):
    enc_hashes = protect_hashes(hashes, secret)
    payload = {
        "hashes": enc_hashes,
        "pad_lens": pad_lens,
        "config": config,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_hash_sidecar(path, secret):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    hashes = unprotect_hashes(payload["hashes"], secret)
    pad_lens = list(payload["pad_lens"])
    config = dict(payload["config"])
    return hashes, pad_lens, config




def split_into_chunks(samples, chunk_size):
    return [samples[i: i + chunk_size] for i in range(0, len(samples), chunk_size)]

def encrypt_signal(samples, meta, chunk_size, row_len, itr, mode, emd_max_imfs, emd_max_siftings):
    chunks = split_into_chunks(samples, chunk_size)
    enc_chunks: List[np.ndarray] = []
    hashes: List[bytes] = []
    pad_lens: List[int] = []

    i = 0
    for chunk in chunks:
        enc_chunk, h, pad_len = encrypt_chunk(chunk_samples=chunk, meta=meta, row_len=row_len, itr=itr, mode=mode, emd_max_imfs=emd_max_imfs, emd_max_siftings=emd_max_siftings)
        i+=1
        enc_chunks.append(enc_chunk)
        hashes.append(h)
        pad_lens.append(pad_len)

    encrypted = np.concatenate(enc_chunks).astype(np.int64) if enc_chunks else np.array([], dtype=np.int64)
    return encrypted, hashes, pad_lens

def decrypt_signal(samples, hashes, pad_lens, meta, chunk_size, row_len, itr, mode, emd_max_imfs, emd_max_siftings):
    chunks = split_into_chunks(samples, chunk_size)
    if len(chunks) != len(hashes):
        raise ValueError("chunks amount != h_i")

    dec_chunks: List[np.ndarray] = []

    for enc_chunk, h, pad_len in zip(chunks, hashes, pad_lens):
        dec_chunk = decrypt_chunk(enc_chunk_samples=enc_chunk, h=h, meta=meta, row_len=row_len, itr=itr, mode=mode, pad_len=pad_len, emd_max_imfs=emd_max_imfs, emd_max_siftings=emd_max_siftings)
        dec_chunks.append(dec_chunk)

    decrypted = np.concatenate(dec_chunks).astype(np.int64) if dec_chunks else np.array([], dtype=np.int64)
    return decrypted

def test_5_run(org, enc, dec, errored_enc=None, org_format=None, new_format=None):
    sidecar_out = "chunk_hashes.json"
    chunk_size = 16384
    row_len = 8192
    itr = 1
    mode = "modadd"
    emd_max_imfs = 6
    emd_max_siftings = 8
    hash_secret = "secret"

    samples, meta = read_wav_mono(org)

    encrypted, hashes, pad_lens = encrypt_signal(samples, meta, chunk_size, row_len, itr, mode, emd_max_imfs, emd_max_siftings)
    write_wav_mono(enc, encrypted, meta)
    if not errored_enc==None:
        me.make_err(enc, errored_enc)
        enc = errored_enc
    if(org_format != None and new_format != None):
        me.emulate_format_convertion(enc, org_format, new_format)

    save_hash_sidecar(sidecar_out, hashes, pad_lens, hash_secret, config = {
            "chunk_size": chunk_size,
            "row_len": row_len,
            "itr": itr,
            "mode": mode,
            "emd_max_imfs": emd_max_imfs,
            "emd_max_siftings": emd_max_siftings,
    })

    enc_samples, enc_meta = read_wav_mono(enc)
    hashes2, pad_lens2, cfg = load_hash_sidecar(sidecar_out, hash_secret)

    decrypted = decrypt_signal(enc_samples, hashes2, pad_lens2, enc_meta, cfg["chunk_size"], cfg["row_len"], cfg["itr"], cfg["mode"], cfg["emd_max_imfs"], cfg["emd_max_siftings"])
    write_wav_mono(dec, decrypted, enc_meta)

    # diff = samples.astype(np.int64) - decrypted.astype(np.int64)


