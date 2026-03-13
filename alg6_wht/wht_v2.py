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

CHUNK_SIZE = 4096

KEEP_COEFFS = 2048

META_SECRET = "very_secret_shared_key"

# modified Henon map parameters 
HENON_A = 3.58
HENON_B = 0.56

# hyperchaotic Lorenz parameters f
LZ_A = 10.0
LZ_B = 8.0 / 3.0
LZ_C = 28.0
LZ_KP = -3.6
LZ_KI = 5.2

# сколько первых значений выкидываем
HENON_BURN_IN = 1000
LORENZ_BURN_IN = 1000


_PERM_CACHE = {}

def derive_key1_key2(secret: str, chunk_index: int) -> Tuple[float, float]:
    h = hashlib.sha3_512()
    h.update(secret.encode("utf-8"))
    h.update(chunk_index.to_bytes(8, "big"))
    digest = h.digest()

    a = int.from_bytes(digest[:8], "big") / float(2**64)
    b = int.from_bytes(digest[8:16], "big") / float(2**64)

    eps = 1e-15
    a = min(max(a, eps), 1.0 - eps)
    b = min(max(b, eps), 1.0 - eps)

    return float(a), float(b)



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



def pcm_to_centered_signed(samples_pcm: np.ndarray, meta: Dict) -> np.ndarray:
    if meta["sample_width_bytes"] == 1:
        return samples_pcm.astype(np.int64) - 128
    return samples_pcm.astype(np.int64)


def centered_signed_to_pcm(samples_signed: np.ndarray, meta: Dict) -> np.ndarray:
    if meta["sample_width_bytes"] == 1:
        return (samples_signed.astype(np.int64) + 128)
    return samples_signed.astype(np.int64)


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

def derive_henon_init(secret: str, chunk_index: int) -> Tuple[float, float]:
    h1 = hashlib.sha3_512()
    h1.update(secret.encode("utf-8"))
    h1.update(chunk_index.to_bytes(8, "big"))
    h1.update(b"henon_x0")
    d1 = h1.digest()

    h2 = hashlib.sha3_512()
    h2.update(secret.encode("utf-8"))
    h2.update(chunk_index.to_bytes(8, "big"))
    h2.update(b"henon_y0")
    d2 = h2.digest()

    x_raw = int.from_bytes(d1[:8], "big") / float(2**64)
    y_raw = int.from_bytes(d2[:8], "big") / float(2**64)

    eps = 1e-15
    x0 = min(max(x_raw, eps), 1.0 - eps)
    y0 = min(max(y_raw, eps), 1.0 - eps)
    return x0, y0


def build_permutation(length: int, chunk_index: int) -> np.ndarray:
    henon_x0, henon_y0 = derive_henon_init(META_SECRET, chunk_index)

    key = (length, chunk_index, HENON_A, HENON_B, henon_x0, henon_y0, HENON_BURN_IN)
    if key in _PERM_CACHE:
        return _PERM_CACHE[key]

    total_points = length
    xs, ys = generate_modified_henon_sequence_jit(
        total_points=total_points,
        a=HENON_A,
        b=HENON_B,
        x0=henon_x0,
        y0=henon_y0,
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

# SUBKEYS и INITIAL CONDITIONS

def xor_float32(a: float, b: float) -> float:
    a32 = np.float32(a)
    b32 = np.float32(b)

    a_bytes = a32.tobytes()
    b_bytes = b32.tobytes()

    out = bytes(x ^ y for x, y in zip(a_bytes, b_bytes))
    return np.frombuffer(out, dtype=np.float32)[0].item()


def normalize_to_unit_interval(x: float, eps: float = 1e-15) -> float:
    v = abs(x) - math.floor(abs(x))
    if v < eps:
        v = eps
    if v > 1.0 - eps:
        v = 1.0 - eps
    return v


def derive_initial_conditions_from_key1_key2(key1: float, key2: float) -> Tuple[float, float, float, float]:
    s1_raw = (key1 + key2) % 1.0
    s2_raw = key1 - key2
    s3_raw = key1 * key2
    s4_raw = xor_float32(key1, key2)

    x0 = normalize_to_unit_interval(s1_raw)
    y0 = normalize_to_unit_interval(s2_raw)
    z0 = normalize_to_unit_interval(s3_raw)
    w0 = normalize_to_unit_interval(s4_raw)

    return x0, y0, z0, w0

_KEYNORM_CACHE = {}

@njit(cache=True, fastmath=True)
def lorenz_hyperchaotic_step_jit(
    x: float,
    y: float,
    z: float,
    w: float,
    a: float,
    b: float,
    c: float,
    kp: float,
    ki: float,
):
    dx = a * (y - x) + w
    dy = c * x - y - x * z
    dz = x * y - b * z
    dw = ki * a * (y - x) + kp * x
    return dx, dy, dz, dw


@njit(cache=True, fastmath=True)
def rk4_lorenz_hyperchaotic_jit(
    x0: float,
    y0: float,
    z0: float,
    w0: float,
    n: int,
    h: float,
    a: float,
    b: float,
    c: float,
    kp: float,
    ki: float,
):
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    zs = np.empty(n, dtype=np.float64)
    ws = np.empty(n, dtype=np.float64)

    x = x0
    y = y0
    z = z0
    w = w0

    for i in range(n):
        xs[i] = x
        ys[i] = y
        zs[i] = z
        ws[i] = w

        k1x, k1y, k1z, k1w = lorenz_hyperchaotic_step_jit(x, y, z, w, a, b, c, kp, ki)

        k2x, k2y, k2z, k2w = lorenz_hyperchaotic_step_jit(
            x + h * k1x / 2.0,
            y + h * k1y / 2.0,
            z + h * k1z / 2.0,
            w + h * k1w / 2.0,
            a, b, c, kp, ki,
        )

        k3x, k3y, k3z, k3w = lorenz_hyperchaotic_step_jit(
            x + h * k2x / 2.0,
            y + h * k2y / 2.0,
            z + h * k2z / 2.0,
            w + h * k2w / 2.0,
            a, b, c, kp, ki,
        )

        k4x, k4y, k4z, k4w = lorenz_hyperchaotic_step_jit(
            x + h * k3x,
            y + h * k3y,
            z + h * k3z,
            w + h * k3w,
            a, b, c, kp, ki,
        )

        x = x + h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        y = y + h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        z = z + h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
        w = w + h * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0

    return xs, ys, zs, ws

def build_key_norm_from_key1_key2(
    key1: float,
    key2: float,
    length: int,
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

    X = (xi ^ yi ^ zi ^ wi).astype(np.float64)

    xmin = float(np.min(X))
    xmax = float(np.max(X))

    if abs(xmax - xmin) < 1e-18:
        key_norm = np.zeros(len(X), dtype=np.float32)
    else:
        key_norm = ((X - xmin) / (xmax - xmin)).astype(np.float32)

    return key_norm, (x0, y0, z0, w0)

@njit(cache=True, fastmath=True)
def hyperchaotic_value_to_integer_sequence_jit(arr: np.ndarray) -> np.ndarray:
    out = np.empty(len(arr), dtype=np.uint64)
    for i in range(len(arr)):
        v = arr[i]
        frac = abs(v) - math.floor(abs(v))
        val = frac * 1e14
        out[i] = np.uint64(int(math.floor(val)))
    return out

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
    out = np.empty(len(arr), dtype=np.uint64)
    for i, v in enumerate(arr):
        frac = abs(v) - math.floor(abs(v))
        val = frac * 1e14
        out[i] = np.uint64(int(math.floor(val)))
    return out


def xor_float32_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)

    if len(a32) != len(b32):
        raise ValueError("Length mismatch in xor_float32_arrays")

    a_bytes = a32.view(np.uint8).reshape(-1, 4)
    b_bytes = b32.view(np.uint8).reshape(-1, 4)
    out_bytes = np.bitwise_xor(a_bytes, b_bytes)

    return out_bytes.reshape(-1).view(np.float32)


def float32_array_to_int32_container(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32).view(np.int32).astype(np.int64)


def int32_container_to_float32_array(arr: np.ndarray) -> np.ndarray:
    arr32 = arr.astype(np.int32)
    return arr32.view(np.float32)

# ШИФРОВАНИЕ / ДЕШИФРОВАНИЕ ЧАНКА
def encrypt_chunk(chunk_signed: np.ndarray, meta: Dict, chunk_index: int) -> np.ndarray:
    n = len(chunk_signed)
    if not is_power_of_two(n):
        raise ValueError("chunk size must be power of two for FWHT")

    if not (1 <= KEEP_COEFFS <= n):
        raise ValueError("KEEP_COEFFS must be in range 1..chunk_size")

    # FWHT
    mw = fwht_inplace_float(chunk_signed.astype(np.float64))

    # zero higher order coefficients
    mwz = mw.copy()
    if KEEP_COEFFS < n:
        mwz[KEEP_COEFFS:] = 0.0

    # key1/key2 from compresed signal
    key1, key2 = derive_key1_key2(META_SECRET, chunk_index)

    # permutatio Henon
    perm = build_permutation(len(mwz), chunk_index)
    mwz_perm = mwz[perm]

    # key_norm Lorenz hyperchaos
    key_norm, init_cond = build_key_norm_from_key1_key2(
    key1=key1,
    key2=key2,
    length=len(mwz_perm),
)

    mwz_perm_f32 = mwz_perm.astype(np.float32)
    cipher_f32 = xor_float32_arrays(mwz_perm_f32, key_norm)

    cipher_i32 = float32_array_to_int32_container(cipher_f32)

    return cipher_i32.astype(np.int64)


def decrypt_chunk(cipher_signed: np.ndarray, meta: Dict, chunk_index: int) -> np.ndarray:
    #ciphertext cont
    cipher_f32 = int32_container_to_float32_array(cipher_signed)
    key1,key2 = derive_key1_key2(META_SECRET, chunk_index)
    #regenerate key_norm
    key_norm, _ = build_key_norm_from_key1_key2(
    key1=key1,
    key2=key2,
    length=len(cipher_f32),
)

    # xor back
    mwz_perm_f32 = xor_float32_arrays(cipher_f32, key_norm)

    # 4) inverse permutation
    perm = build_permutation(len(mwz_perm_f32), chunk_index)
    inv = invert_permutation(perm)
    mwz_rec = mwz_perm_f32[inv].astype(np.float64)

    # inverse FWHT
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

def encrypt_signal(samples_pcm: np.ndarray, meta: Dict) -> np.ndarray:
    samples_signed = pcm_to_centered_signed(samples_pcm, meta)
    chunks, original_total = split_into_fixed_chunks_with_pad(samples_signed, CHUNK_SIZE)

    enc_chunks: List[np.ndarray] = []
    debug_info: List[Dict] = []

    for idx, chunk in enumerate(chunks):
        enc_chunk_signed = encrypt_chunk(chunk, meta, idx)

        enc_chunks.append(enc_chunk_signed)

    encrypted_signed = np.concatenate(enc_chunks).astype(np.int64)
    return encrypted_signed


def decrypt_signal(encrypted_pcm: np.ndarray, meta: Dict) -> np.ndarray:

    encrypted_signed = encrypted_pcm.astype(np.int64)

    chunks = split_exact_chunks(encrypted_signed, CHUNK_SIZE)

    dec_chunks: List[np.ndarray] = []

    for idx, cipher_chunk in enumerate(chunks):

        dec_chunk_signed = decrypt_chunk(
            cipher_signed=cipher_chunk,
            meta=meta,
            chunk_index=idx,
        )

        dec_chunks.append(dec_chunk_signed)

    decrypted_signed = np.concatenate(dec_chunks).astype(np.int64)

    decrypted_pcm = centered_signed_to_pcm(decrypted_signed, meta)

    return decrypted_pcm


def build_cipher_meta_from_plain_meta(meta: Dict) -> Dict:
    return {
        "sample_rate": meta["sample_rate"],
        "sample_width_bytes": 4,
        "bits_per_sample": 32,
        "min_value": -(1 << 31),
        "max_value": (1 << 31) - 1,
        "dtype_name": "int32",
    }


def test_6_run(org, enc, dec, errored_enc=None, org_format=None, new_format=None):
    samples_pcm, meta = read_wav_mono(org)

    encrypted_container = encrypt_signal(samples_pcm, meta)

    cipher_meta = build_cipher_meta_from_plain_meta(meta)

    write_wav_mono(enc, encrypted_container, cipher_meta)

    enc_pcm_2, enc_meta = read_wav_mono(enc)

    decrypted_pcm = decrypt_signal(enc_pcm_2, meta)

    write_wav_mono(dec, decrypted_pcm, meta)
