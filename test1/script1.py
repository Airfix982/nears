import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write as write_wav
import script.utils as me


# =========================
# Chaotic map + PRBG (ключевой поток)
# =========================

def map_fun(x: float, r: float) -> float:
    """Chaotic map iteration: x_{n+1} = r*sin(pi*x_n) + r*(10^3*x_n mod 1)"""
    return r * np.sin(np.pi * x) + r * np.remainder(10 ** 3 * x, 1)


def trajectory(x0: float, r: float, num_points: int) -> list[float]:
    """Generate num_points values of chaotic trajectory starting from x0."""
    points = []
    x = x0
    for _ in range(num_points):
        x = map_fun(x, r)
        points.append(x)
    return points


def prbg(x: float) -> int:
    """Pseudo-random bit generator from a float."""
    return int(np.mod(10 ** 10 * x, 1) > 0.5)


def random_bits(traj: list[float]) -> list[int]:
    """Convert trajectory values to bits."""
    return list(map(prbg, traj))


def random_bits_to_int16(bits: list[int]) -> np.ndarray:
    bits = np.array(bits, dtype=np.uint8)

    # reshape в (num_samples, 16)
    bits = bits.reshape(-1, 16)

    # создаём веса битов
    powers = (1 << np.arange(15, -1, -1)).astype(np.int32)

    # быстрое матричное умножение
    values = bits.dot(powers)

    # сдвиг в signed диапазон
    values = values - 2**15

    return values.astype(np.int16)


def generate_key_stream_int16(num_samples: int, x0: float, r: float) -> tuple[np.ndarray, float]:
    """
    Generate key stream (int16) for XOR of length num_samples.
    Returns (key, next_x0) where next_x0 is last trajectory value for continuation.
    """
    traj = trajectory(x0, r, 16 * num_samples)   # 16 bits per sample
    next_x0 = traj[-1] if traj else x0
    bits = random_bits(traj)
    key = random_bits_to_int16(bits)             # length == num_samples
    return key[:num_samples], next_x0






def encrypt_wav_file_realtime_like(
    input_wav_path: str,
    output_wav_path: str,
    x0: float,
    r: float,
    block_samples: int
) -> float:
    fs, samples = wavfile.read(input_wav_path)

    samples = np.asarray(samples, dtype=np.int16)

    if samples.ndim == 2:
        samples = samples[:, 0]

    encrypted_blocks = []

    pos = 0
    while pos < len(samples):
        chunk = samples[pos:pos + block_samples]

        key, x0 = generate_key_stream_int16(len(chunk), x0, r)

        enc = np.bitwise_xor(chunk, key).astype(np.int16)
        encrypted_blocks.append(enc)

        pos += len(chunk)

    encrypted = np.concatenate(encrypted_blocks)
    write_wav(output_wav_path, fs, encrypted)
    return x0


def decrypt_wav_file_realtime_like(
    input_wav_path: str,
    output_wav_path: str,
    x0: float,
    r: float,
    block_samples: int
) -> float:
    fs, encrypted = wavfile.read(input_wav_path)

    encrypted = np.asarray(encrypted, dtype=np.int16)
    if encrypted.ndim == 2:
        encrypted = encrypted[:, 0]

    decrypted_blocks = []

    pos = 0
    while pos < len(encrypted):
        chunk = encrypted[pos:pos + block_samples]

        key, x0 = generate_key_stream_int16(len(chunk), x0, r)
        dec = np.bitwise_xor(chunk, key).astype(np.int16)
        decrypted_blocks.append(dec)

        pos += len(chunk)

    decrypted = np.concatenate(decrypted_blocks)
    write_wav(output_wav_path, fs, decrypted)
    return x0



def test_1_run(org, enc, dec, x0=0.1234567890123456, r=0.9876543210987654, errored_enc=None, org_format=None, new_format=None):
    # org = "test1/artefacts/input.wav"
    # enc = "test1/artefacts/encrypted.wav"
    # dec = "test1/artefacts/decrypted.wav"

    fs, _ = wavfile.read(org)
    time_sample = 0.5 # 0.5 second
    block_samples = int(time_sample * fs)

    encrypt_wav_file_realtime_like(org, enc, x0, r, block_samples)
    if not errored_enc==None:
        me.make_err(enc, errored_enc)
        enc = errored_enc
    if(org_format != None and new_format != None):
        me.emulate_format_convertion(enc, org_format, new_format)

    decrypt_wav_file_realtime_like(enc, dec, x0, r, block_samples)




