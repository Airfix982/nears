import time
import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt
import script.utils as me

class ChaosKeys:

  def __init__(self, min_key_length, max_key_length, mod, secret1, secret2, secret3):

    # Decide on gamma and find etas
    self.gamma = 31 - 27 * secret1 / mod
    self.eta1 = 0.5 - np.sqrt(0.25 - np.floor(self.gamma / 4) / self.gamma)
    self.eta2 = 0.5 + np.sqrt(0.25 - np.floor(self.gamma / 4) / self.gamma)

    # Decide on x
    self.x = secret2 / mod
    self.keys = []

    # Decide on key list length
    if min_key_length != max_key_length:
      self.key_length = min_key_length + secret3 % (max_key_length - min_key_length)
    else:
      self.key_length = min_key_length

  # Use chaos maps to generate keys
  # Read into research article (at the top) for more info
  def generage_keys(self):
    bytestream = []
    bitcount = 8

    # Slightly modified to output bytes instead of bits
    while len(bytestream) < self.key_length:
      if self.eta1 <= self.x <= self.eta2:
        self.x = (self.gamma * self.x * (1 - self.x) % 1) / (self.gamma / 4 % 1)
      else:
        self.x = self.gamma * self.x * (1 - self.x) % 1
      if 0.1 <= self.x <= 0.6:
        bit = 1 if self.x * 10**10 % 1 > 0.5 else 0
        if bitcount < 8:
          bitcount += 1
          bytestream[-1] = bytestream[-1] * 2 + bit
        else:
          bitcount = 1
          bytestream.append(bit)

    self.keys = bytestream
    return self.keys

  # Plot key values
  def plot_keys(self):
    plt.plot(self.keys, "o")
    plt.xlabel("Keys Index")
    plt.ylabel("Key Value")
    plt.show()



# ====== параметры как в репо ======
RATE = 48000
BLOCKSIZE = 2400  # 2400 @ 48kHz = 50ms блок


# ====== XOR и обёртка ключей (как в репо) ======
def wrap_keys(xor_keys: np.ndarray, curr_key_idx: int, need_bytes: int) -> tuple[bytes, int]:
    # xor_keys: uint8
    keys = xor_keys[curr_key_idx:]
    while keys.size < need_bytes:
        keys = np.concatenate((keys, xor_keys), 0)

    curr_key_idx = (curr_key_idx + need_bytes) % xor_keys.size
    keys = keys[:need_bytes]
    return keys.tobytes(), curr_key_idx


def byte_xor(a: bytes, b: bytes) -> bytes:
    # математически то же самое, что int.from_bytes(...) ^ int.from_bytes(...),
    # только быстрее и без гигантских big-int.
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()


# ====== приведение WAV к формату, который ожидает их пайп ======
def wav_to_int32_mono(samples: np.ndarray) -> np.ndarray:
    # В репо при чтении файла они сводят в моно средним и приводят к int32
    if samples.ndim == 2:
        samples = np.average(samples, axis=1)

    # Дальше — их ветки из audio_record.py (mic_input=False)
    if samples.dtype == np.float32:
        samples = np.array(samples * 2147483647, dtype=np.int32)
    elif samples.dtype == np.int16:
        samples = samples.astype(np.int32) * 65538
    elif samples.dtype == np.uint8:
        samples = samples.astype(np.int32) * 16777216 - 2147483648
    else:
        # если уже int32 — оставляем
        samples = np.array(samples, dtype=np.int32)

    # Важно: в sounddevice callback у них форма (blocksize, 1)
    return samples.reshape((-1, 1))


# ====== шифр/дешифр блоками “как realtime” ======
def encrypt_wav_realtime_like(input_wav: str, encrypted_wav: str, xor_keys: np.ndarray,
                              emulate_realtime: bool = False) -> None:
    fs, samples = wavfile.read(input_wav)
    if fs != RATE:
        print(f"[warn] input rate={fs}, orig rate={RATE}")

    data = wav_to_int32_mono(samples)

    curr_key_idx = 0
    out_blocks = []

    wavelet = pywt.Wavelet("db1")

    pos = 0
    while pos < data.shape[0]:
        t0 = time.perf_counter()

        block = np.zeros((BLOCKSIZE, 1), dtype=np.int32)
        take = min(BLOCKSIZE, data.shape[0] - pos)
        block[:take] = data[pos:pos + take]

        # как в audio_record.py:
        audio_dwt = pywt.dwt(block, wavelet)[0]              # DWT
        audio_dwt = np.array(audio_dwt, dtype=np.int32)      # cast to int32
        plain_bytes = np.ndarray.tobytes(audio_dwt)

        key_bytes, curr_key_idx = wrap_keys(xor_keys, curr_key_idx, BLOCKSIZE * 4)
        enc_bytes = byte_xor(plain_bytes, key_bytes)

        enc_i32 = np.frombuffer(enc_bytes, dtype=np.int32).reshape((BLOCKSIZE, 1))
        out_blocks.append(enc_i32)

        pos += take

        if emulate_realtime:
            # подгоняем, чтобы обработка “шла” как реальное время блока
            dt = time.perf_counter() - t0
            target = BLOCKSIZE / float(fs)
            if dt < target:
                time.sleep(target - dt)

    encrypted = np.vstack(out_blocks).astype(np.int32)
    wavfile.write(encrypted_wav, fs, encrypted)


def decrypt_wav_realtime_like(encrypted_wav: str, decrypted_wav: str, xor_keys: np.ndarray,
                              emulate_realtime: bool = False) -> None:
    fs, enc = wavfile.read(encrypted_wav)
    enc = wav_to_int32_mono(enc)  # приведём к (N,1) int32

    curr_key_idx = 0
    out_blocks = []

    wavelet = pywt.Wavelet("db1")

    pos = 0
    while pos < enc.shape[0]:
        t0 = time.perf_counter()

        block = np.zeros((BLOCKSIZE, 1), dtype=np.int32)
        take = min(BLOCKSIZE, enc.shape[0] - pos)
        block[:take] = enc[pos:pos + take]

        enc_bytes = np.ndarray.tobytes(block)

        key_bytes, curr_key_idx = wrap_keys(xor_keys, curr_key_idx, BLOCKSIZE * 4)
        plain_bytes = byte_xor(enc_bytes, key_bytes)

        wavelet_data = np.frombuffer(plain_bytes, dtype=np.int32).reshape((BLOCKSIZE, 1))

        # как в audio_play.py:
        audio = pywt.idwt(wavelet_data, None, wavelet)

# как у них "average", но НЕ убиваем дробь в ноль
        audio = np.average(audio, axis=1)  # float

        audio = audio.reshape((BLOCKSIZE, 1)).astype(np.float32)

        out_blocks.append(audio)

        pos += take

        if emulate_realtime:
            dt = time.perf_counter() - t0
            target = BLOCKSIZE / float(fs)
            if dt < target:
                time.sleep(target - dt)

    decrypted = np.vstack(out_blocks).astype(np.float32)
    m = np.max(np.abs(decrypted))
    if m > 0:
        decrypted_float_norm = decrypted / m
    else:
        decrypted_float_norm = decrypted

    # Пишем int16
    decrypted_i16 = (decrypted_float_norm * 32767.0).astype(np.int16)
    wavfile.write(decrypted_wav, fs, decrypted_i16)




def test_2_run(org, enc, dec, secret1 = 11111, secret2 = 22222, secret3 = 33333, errored_enc=None, org_format=None, new_format=None):
    # org = "test2/artefacts/input.wav"
    # enc = "test2/artefacts/encrypted.wav"
    # dec = "test2/artefacts/decrypted.wav"

    
    # ==== как получить xor_keys без DH ====
    # Ты просил “оставить только шифрование”.
    # Значит DH выкидываем, а секреты задаём константами (или потом подставишь свои).
    mod = 2**16 + 1
    min_len = 16384
    max_len = 32768

    xor_keys = np.array(
        ChaosKeys(min_len, max_len, mod, secret1, secret2, secret3).generage_keys(),
        dtype=np.uint8
    )
    emulate_real_time = True

    encrypt_wav_realtime_like(org, enc, xor_keys, emulate_realtime=emulate_real_time)
    if not errored_enc==None:
        me.make_err(enc, errored_enc)
        enc = errored_enc
    if(org_format != None and new_format != None):
        me.emulate_format_convertion(enc, org_format, new_format)
    decrypt_wav_realtime_like(enc, dec, xor_keys, emulate_realtime=emulate_real_time)
    print("done")
