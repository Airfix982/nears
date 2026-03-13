import wave
import json
import math
import base64
import struct
import hashlib
from typing import Dict, List, Tuple

import numpy as np
from numba import njit

SIDECAR_JSON = "chunk_meta.json"

# realtime chunk size in SAMPLES
CHUNK_SIZE = 4096

# сколько FWHT-коэффициентов оставить, остальные занулить
KEEP_COEFFS = 2048

# секрет для защиты метаинформации чанков
META_SECRET = "very_secret_shared_key"

# modified Henon map parameters from article
HENON_A = 3.58
HENON_B = 0.56

# fixed initial conditions for Henon permutation
HENON_X0 = 0.1
HENON_Y0 = 0.1

# hyperchaotic Lorenz parameters from article
LZ_A = 10.0
LZ_B = 8.0 / 3.0
LZ_C = 28.0
LZ_KP = -3.6
LZ_KI = 5.2

# сколько первых значений выкидываем
HENON_BURN_IN = 1000
LORENZ_BURN_IN = 1000


_PERM_CACHE = {}


# WAV I/O

def read_wav_mono(path: str) -> Tuple[np.ndarray, Dict]:
    with wave.open(path, "rb") as f:
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        n_frames = f.getnframes()
        comptype = f.getcomptype()
        raw = f.readframes(n_frames)

    if comptype != "NONE":
        raise ValueError("Only uncompressed PCM WAV is supported")
    if n_channels != 1:
        raise ValueError("Only mono WAV is supported")

    bits = 8 * sampwidth

    if sampwidth == 1:
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
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    meta = {
        "sample_rate": framerate,
        "sample_width_bytes": sampwidth,
        "bits_per_sample": bits,
        "min_value": min_val,
        "max_value": max_val,
        "dtype_name": dtype_name,
    }
    return arr, meta


def clip_to_pcm_range(samples: np.ndarray, meta: Dict) -> np.ndarray:
    minimum = meta["min_value"]
    maximum = meta["max_value"]
    return np.clip(np.rint(samples), minimum, maximum).astype(np.int64)


def write_wav_mono(path: str, samples_pcm: np.ndarray, meta: Dict) -> None:
    samples_pcm = clip_to_pcm_range(samples_pcm, meta)

    sampwidth = meta["sample_width_bytes"]
    framerate = meta["sample_rate"]

    if sampwidth == 1:
        raw = samples_pcm.astype(np.uint8).tobytes()
    elif sampwidth == 2:
        raw = samples_pcm.astype("<i2").tobytes()
    elif sampwidth == 4:
        raw = samples_pcm.astype("<i4").tobytes()
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.writeframes(raw)


# ПРЕОБРАЗОВАНИЕ PCM <-> ЦЕНТРИРОВАННЫЕ SIGNED WORDS

def pcm_to_centered_signed(samples_pcm: np.ndarray, meta: Dict) -> np.ndarray:
    if meta["sample_width_bytes"] == 1:
        return samples_pcm.astype(np.int64) - 128
    return samples_pcm.astype(np.int64)


def centered_signed_to_pcm(samples_signed: np.ndarray, meta: Dict) -> np.ndarray:
    if meta["sample_width_bytes"] == 1:
        return (samples_signed.astype(np.int64) + 128)
    return samples_signed.astype(np.int64)


def signed_domain_limits(bits: int) -> Tuple[int, int]:
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


# БАЙТОВЫЕ WORD ОПЕРАЦИИ

def signed_to_unsigned_words(values: np.ndarray, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    return (values.astype(np.int64) & mask).astype(np.uint64)


def unsigned_to_signed_words(values: np.ndarray, bits: int) -> np.ndarray:
    sign_bit = 1 << (bits - 1)
    full = 1 << bits
    v = values.astype(np.int64)
    return np.where(v >= sign_bit, v - full, v).astype(np.int64)


#FWHT/IFWHT

def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def fwht_inplace_float(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64).copy()
    n = len(y)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = y[i:i + h].copy()
            b = y[i + h:i + 2 * h].copy()
            y[i:i + h] = a + b
            y[i + h:i + 2 * h] = a - b
        h *= 2
    return y


def ifwht_inplace_float(x: np.ndarray) -> np.ndarray:
    n = len(x)
    return fwht_inplace_float(x) / n


# CHUNKING
# Последний чанк добиваем нулями до полного CHUNK_SIZE
# После дешифровки обрезаем до original_total_samples

def split_into_fixed_chunks_with_pad(samples: np.ndarray, chunk_size: int) -> Tuple[List[np.ndarray], int]:
    original_total = len(samples)
    pad_len = (-len(samples)) % chunk_size
    if pad_len:
        samples = np.pad(samples, (0, pad_len), mode="constant", constant_values=0)

    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
    return chunks, original_total


def split_exact_chunks(samples: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    if len(samples) % chunk_size != 0:
        raise ValueError("Encrypted signal length must be multiple of chunk_size")
    return [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]


# HENON MAP PERMUTATION
# В статье не дан алгоритм построения биективной permutation matrix из хаотической траектории. Поэтому здесь юзаю детерминированная и обратимую интерпретацию: строим длинную траекторию, берём последние P значений, ранжируем их и получаем permutation

def generate_modified_henon_sequence(
    total_points: int,
    a: float,
    b: float,
    x0: float,
    y0: float,
    burn_in: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total = total_points + burn_in
    xs = np.empty(total, dtype=np.float64)
    ys = np.empty(total, dtype=np.float64)

    xs[0] = x0
    ys[0] = y0

    for i in range(total - 1):
        xn = xs[i]
        yn = ys[i]
        xs[i + 1] = 1.0 - a * math.cos(xn) - b * yn
        ys[i + 1] = -xn

    return xs[burn_in:], ys[burn_in:]


def build_permutation(length: int) -> np.ndarray:
    key = (length, HENON_A, HENON_B, HENON_X0, HENON_Y0, HENON_BURN_IN)

    if key in _PERM_CACHE:
        return _PERM_CACHE[key]

    total_points = length * length 
    xs, ys = generate_modified_henon_sequence_jit(
        total_points=total_points,
        a=HENON_A,
        b=HENON_B,
        x0=HENON_X0,
        y0=HENON_Y0,
        burn_in=HENON_BURN_IN,
    )

    xs_tail = xs[-length:]
    ys_tail = ys[-length:]
    scores = xs_tail + ys_tail

    perm = np.argsort(scores, kind="mergesort").astype(np.int64)
    _PERM_CACHE[key] = perm
    return perm


def invert_permutation(permutation: np.ndarray) -> np.ndarray:
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(permutation), dtype=np.int64)
    return inv


# KEY1/KEY2

def compute_key1_key2(mw: np.ndarray) -> Tuple[float, float]:
    even_vals = mw[0::2]
    odd_vals = mw[1::2]

    key1 = float(np.mean(odd_vals)) if len(odd_vals) else 0.0
    key2 = float(np.mean(even_vals)) if len(even_vals) else 0.0
    return key1, key2


# SUBKEYS и INITIAL CONDITIONS
# В статье subkey(4) = key(1) ^ key(2), но key1/key2 вещественные.
# Здесь интерпретация: для XOR берём integerized fractional representation, затем назад к float

def fractional_to_int14(x: float) -> int:
    frac = abs(x) - math.floor(abs(x))
    return int(math.floor(frac * 1e14))


def normalize_to_unit_interval(x: float, eps: float = 1e-15) -> float:
    v = abs(x) - math.floor(abs(x))
    if v < eps:
        v = eps
    if v > 1.0 - eps:
        v = 1.0 - eps
    return v


def derive_initial_conditions_from_key1_key2(key1: float, key2: float) -> Tuple[float, float, float, float]:
    # subkey(1) = key1 + key2, mod 1
    s1_raw = (key1 + key2) % 1.0

    # subkey(2) = key1 - key2
    s2_raw = key1 - key2

    # subkey(3) = key1 * key2
    s3_raw = key1 * key2

    # subkey(4) = key1 ^ key2
    k1i = fractional_to_int14(key1)
    k2i = fractional_to_int14(key2)
    s4_raw = float(k1i ^ k2i) / 1e14

    x0 = normalize_to_unit_interval(s1_raw)
    y0 = normalize_to_unit_interval(s2_raw)
    z0 = normalize_to_unit_interval(s3_raw)
    w0 = normalize_to_unit_interval(s4_raw)

    return x0, y0, z0, w0


# HYPERCHAOTIC LORENZ + RK4

def lorenz_hyperchaotic_step(
    x: float,
    y: float,
    z: float,
    w: float,
    a: float,
    b: float,
    c: float,
    kp: float,
    ki: float,
) -> Tuple[float, float, float, float]:
    dx = a * (y - x) + w
    dy = c * x - y - x * z
    dz = x * y - b * z
    dw = ki * a * (y - x) + kp * x
    return dx, dy, dz, dw


def rk4_lorenz_hyperchaotic(
    x0: float,
    y0: float,
    z0: float,
    w0: float,
    n: int,
    h: float = 0.005,
    a: float = LZ_A,
    b: float = LZ_B,
    c: float = LZ_C,
    kp: float = LZ_KP,
    ki: float = LZ_KI,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    zs = np.empty(n, dtype=np.float64)
    ws = np.empty(n, dtype=np.float64)

    x, y, z, w = x0, y0, z0, w0

    for i in range(n):
        xs[i] = x
        ys[i] = y
        zs[i] = z
        ws[i] = w

        k1 = lorenz_hyperchaotic_step(x, y, z, w, a, b, c, kp, ki)

        k2 = lorenz_hyperchaotic_step(
            x + h * k1[0] / 2,
            y + h * k1[1] / 2,
            z + h * k1[2] / 2,
            w + h * k1[3] / 2,
            a, b, c, kp, ki,
        )

        k3 = lorenz_hyperchaotic_step(
            x + h * k2[0] / 2,
            y + h * k2[1] / 2,
            z + h * k2[2] / 2,
            w + h * k2[3] / 2,
            a, b, c, kp, ki,
        )

        k4 = lorenz_hyperchaotic_step(
            x + h * k3[0],
            y + h * k3[1],
            z + h * k3[2],
            w + h * k3[3],
            a, b, c, kp, ki,
        )

        x = x + h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        y = y + h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        z = z + h * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        w = w + h * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6

    return xs, ys, zs, ws


def hyperchaotic_value_to_integer_sequence(arr: np.ndarray) -> np.ndarray:
    # исправление статьи: mod 1 убран, иначе XOR невозможен
    out = np.empty(len(arr), dtype=np.uint64)
    for i, v in enumerate(arr):
        frac = abs(v) - math.floor(abs(v))
        val = frac * 1e14
        out[i] = np.uint64(int(math.floor(val)))
    return out


def build_keystream_words_from_key1_key2(
    key1: float,
    key2: float,
    length: int,
    bits_per_sample: int,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    x0, y0, z0, w0 = derive_initial_conditions_from_key1_key2(key1, key2)

    total = length + LORENZ_BURN_IN
    xs, ys, zs, ws = rk4_lorenz_hyperchaotic(x0, y0, z0, w0, total)

    xs = xs[LORENZ_BURN_IN:]
    ys = ys[LORENZ_BURN_IN:]
    zs = zs[LORENZ_BURN_IN:]
    ws = ws[LORENZ_BURN_IN:]

    xi = hyperchaotic_value_to_integer_sequence(xs)
    yi = hyperchaotic_value_to_integer_sequence(ys)
    zi = hyperchaotic_value_to_integer_sequence(zs)
    wi = hyperchaotic_value_to_integer_sequence(ws)

    mask = np.uint64((1 << bits_per_sample) - 1)
    ks = (xi ^ yi ^ zi ^ wi) & mask
    return ks, (x0, y0, z0, w0)


# МЕТАДАННЫЕ ЧАНКОВ key1, key2, scale. Защита их симметричным секретом через SHA3-derived keystream XOR.

def derive_meta_keystream(secret_bytes: bytes, counter: int, length: int) -> bytes:
    buf = bytearray()
    block_no = 0
    while len(buf) < length:
        h = hashlib.sha3_512()
        h.update(secret_bytes)
        h.update(counter.to_bytes(8, "big"))
        h.update(block_no.to_bytes(8, "big"))
        buf.extend(h.digest())
        block_no += 1
    return bytes(buf[:length])


def pack_chunk_meta(key1: float, key2: float, scale: float) -> bytes:
    return struct.pack("<ddd", key1, key2, scale)


def unpack_chunk_meta(blob: bytes) -> Tuple[float, float, float]:
    return struct.unpack("<ddd", blob)


def protect_chunk_meta(key1: float, key2: float, scale: float, secret: str, chunk_index: int) -> str:
    raw = pack_chunk_meta(key1, key2, scale)
    ks = derive_meta_keystream(secret.encode("utf-8"), chunk_index, len(raw))
    enc = bytes(a ^ b for a, b in zip(raw, ks))
    return base64.b64encode(enc).decode("ascii")


def unprotect_chunk_meta(encoded: str, secret: str, chunk_index: int) -> Tuple[float, float, float]:
    enc = base64.b64decode(encoded.encode("ascii"))
    ks = derive_meta_keystream(secret.encode("utf-8"), chunk_index, len(enc))
    raw = bytes(a ^ b for a, b in zip(enc, ks))
    return unpack_chunk_meta(raw)


def save_sidecar(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_sidecar(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# КВАНТОВАНИЕ FWHT-КОЭФФИЦИЕНТОВ В WORD DOMAIN

def quantize_transform_coeffs(mw: np.ndarray, bits: int) -> Tuple[np.ndarray, float]:
    smin, smax = signed_domain_limits(bits)
    full_scale = float(max(abs(smin), abs(smax)))
    peak = float(np.max(np.abs(mw))) if len(mw) else 1.0
    scale = peak if peak > 1e-18 else 1.0

    q = np.rint((mw / scale) * full_scale).astype(np.int64)
    q = np.clip(q, smin, smax)
    return q, scale


def dequantize_transform_coeffs(q: np.ndarray, scale: float, bits: int) -> np.ndarray:
    smin, smax = signed_domain_limits(bits)
    full_scale = float(max(abs(smin), abs(smax)))
    return (q.astype(np.float64) / full_scale) * scale


# ШИФРОВАНИЕ / ДЕШИФРОВАНИЕ ЧАНКА

def encrypt_chunk(chunk_signed: np.ndarray, meta: Dict) -> Tuple[np.ndarray, Dict]:
    n = len(chunk_signed)
    if not is_power_of_two(n):
        raise ValueError("chunk size must be power of two for FWHT")

    if not (1 <= KEEP_COEFFS <= n):
        raise ValueError("KEEP_COEFFS must be in range 1..chunk_size")

    # 1) FWHT
    mw = fwht_inplace_float(chunk_signed.astype(np.float64))

    # 2) zero higher order coefficients
    mwz = mw.copy()
    if KEEP_COEFFS < n:
        mwz[KEEP_COEFFS:] = 0.0

    # 3) key1/key2 from compressed signal
    key1, key2 = compute_key1_key2(mwz)

    # 4) quantize coefficients into word domain for XOR transport
    q, scale = quantize_transform_coeffs(mwz, meta["bits_per_sample"])

    # 5) permutation
    perm = build_permutation(len(q))
    q_perm = q[perm]

    # 6) keystream
    ks, init_cond = build_keystream_words_from_key1_key2(
        key1=key1,
        key2=key2,
        length=len(q_perm),
        bits_per_sample=meta["bits_per_sample"],
    )

    # 7) XOR substitution
    q_perm_u = signed_to_unsigned_words(q_perm, meta["bits_per_sample"])
    cipher_u = np.bitwise_xor(q_perm_u, ks)
    cipher_s = unsigned_to_signed_words(cipher_u, meta["bits_per_sample"])

    chunk_meta = {
        "protected": None,  # заполнится снаружи
        "x0": init_cond[0],
        "y0": init_cond[1],
        "z0": init_cond[2],
        "w0": init_cond[3],
    }
    return cipher_s.astype(np.int64), {
        "key1": key1,
        "key2": key2,
        "scale": scale,
        "debug_init": chunk_meta,
    }


def decrypt_chunk(cipher_signed: np.ndarray, meta: Dict, key1: float, key2: float, scale: float) -> np.ndarray:
    bits = meta["bits_per_sample"]

    # 1) regenerate keystream
    ks, _ = build_keystream_words_from_key1_key2(
        key1=key1,
        key2=key2,
        length=len(cipher_signed),
        bits_per_sample=bits,
    )

    # 2) xor back
    cipher_u = signed_to_unsigned_words(cipher_signed, bits)
    q_perm_u = np.bitwise_xor(cipher_u, ks)
    q_perm = unsigned_to_signed_words(q_perm_u, bits)

    # 3) inverse permutation
    perm = build_permutation(len(q_perm))
    inv = invert_permutation(perm)
    q = q_perm[inv]

    # 4) dequantize transform coefficients
    mwz_rec = dequantize_transform_coeffs(q, scale, bits)

    # 5) inverse FWHT
    restored = ifwht_inplace_float(mwz_rec)

    return np.rint(restored).astype(np.int64)


# ШИФРОВАНИЕ ДЕШИФР ВСЕГО СИГНАЛА

@njit(cache=True)
def generate_modified_henon_sequence_jit(
    total_points: int,
    a: float,
    b: float,
    x0: float,
    y0: float,
    burn_in: int,
):
    total = total_points + burn_in
    xs = np.empty(total, dtype=np.float64)
    ys = np.empty(total, dtype=np.float64)

    xs[0] = x0
    ys[0] = y0

    for i in range(total - 1):
        xn = xs[i]
        yn = ys[i]
        xs[i + 1] = 1.0 - a * math.cos(xn) - b * yn
        ys[i + 1] = -xn

    return xs[burn_in:], ys[burn_in:]

def encrypt_signal(samples_pcm: np.ndarray, meta: Dict) -> Tuple[np.ndarray, Dict]:
    samples_signed = pcm_to_centered_signed(samples_pcm, meta)
    print(1)
    chunks, original_total = split_into_fixed_chunks_with_pad(samples_signed, CHUNK_SIZE)
    print(2)

    enc_chunks: List[np.ndarray] = []
    protected_meta_list: List[str] = []
    debug_info: List[Dict] = []
    print(len(chunks))
    for idx, chunk in enumerate(chunks):
        enc_chunk_signed, info = encrypt_chunk(chunk, meta)
        print(idx)

        protected = protect_chunk_meta(
            key1=info["key1"],
            key2=info["key2"],
            scale=info["scale"],
            secret=META_SECRET,
            chunk_index=idx,
        )

        protected_meta_list.append(protected)
        debug_info.append({
            "x0": info["debug_init"]["x0"],
            "y0": info["debug_init"]["y0"],
            "z0": info["debug_init"]["z0"],
            "w0": info["debug_init"]["w0"],
        })

        enc_chunks.append(enc_chunk_signed)

    encrypted_signed = np.concatenate(enc_chunks).astype(np.int64)
    print(4)
    encrypted_pcm = centered_signed_to_pcm(encrypted_signed, meta)
    print(5)

    sidecar = {
        "original_total_samples": original_total,
        "chunk_size": CHUNK_SIZE,
        "keep_coeffs": KEEP_COEFFS,
        "henon_a": HENON_A,
        "henon_b": HENON_B,
        "henon_x0": HENON_X0,
        "henon_y0": HENON_Y0,
        "lorenz_a": LZ_A,
        "lorenz_b": LZ_B,
        "lorenz_c": LZ_C,
        "lorenz_kp": LZ_KP,
        "lorenz_ki": LZ_KI,
        "henon_burn_in": HENON_BURN_IN,
        "lorenz_burn_in": LORENZ_BURN_IN,
        "protected_chunk_meta": protected_meta_list,
        "debug_initial_conditions": debug_info,
    }
    return encrypted_pcm, sidecar


def decrypt_signal(encrypted_pcm: np.ndarray, meta: Dict, sidecar: Dict) -> np.ndarray:
    original_total = int(sidecar["original_total_samples"])
    chunk_size = int(sidecar["chunk_size"])
    protected_meta = sidecar["protected_chunk_meta"]

    encrypted_signed = pcm_to_centered_signed(encrypted_pcm, meta)
    chunks = split_exact_chunks(encrypted_signed, chunk_size)

    if len(chunks) != len(protected_meta):
        raise ValueError("Mismatch: encrypted chunks count != metadata count")

    dec_chunks: List[np.ndarray] = []

    for idx, (cipher_chunk, protected) in enumerate(zip(chunks, protected_meta)):
        key1, key2, scale = unprotect_chunk_meta(
            encoded=protected,
            secret=META_SECRET,
            chunk_index=idx,
        )

        dec_chunk_signed = decrypt_chunk(
            cipher_signed=cipher_chunk,
            meta=meta,
            key1=key1,
            key2=key2,
            scale=scale,
        )
        dec_chunks.append(dec_chunk_signed)

    decrypted_signed = np.concatenate(dec_chunks).astype(np.int64)
    decrypted_signed = decrypted_signed[:original_total]
    decrypted_pcm = centered_signed_to_pcm(decrypted_signed, meta)
    return decrypted_pcm


def test_6_run(org, enc, dec, errored_enc=None, org_format=None, new_format=None):
    print("READ SOURCE WAV")
    samples_pcm, meta = read_wav_mono(org)

    print("ENCRYPT SIGNAL")
    encrypted_pcm, sidecar = encrypt_signal(samples_pcm, meta)

    print("WRITE ENCRYPTED WAV")
    write_wav_mono(enc, encrypted_pcm, meta)

    print("SAVE SIDECAR")
    save_sidecar(SIDECAR_JSON, sidecar)

    print("READ ENCRYPTED WAV")
    enc_pcm_2, enc_meta = read_wav_mono(enc)

    print("LOAD SIDECAR")
    sidecar_2 = load_sidecar(SIDECAR_JSON)

    print("DECRYPT SIGNAL")
    decrypted_pcm = decrypt_signal(enc_pcm_2, enc_meta, sidecar_2)

    print("WRITE DECRYPTED WAV")
    write_wav_mono(dec, decrypted_pcm, enc_meta)

    print("DONE")