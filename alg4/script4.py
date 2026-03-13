from IPython.display import clear_output
import itertools

#To read and write to audio(.wav) files:
import wave

# For Spectogram generation
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

#To clear unnecessary output 
from IPython.display import clear_output

#To get spectogram wave form generation 
import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft

# To play audio files in google colab
import sys
sys.path.insert(0, 'ThinkDSP/code/') 
# import thinkdsp
import matplotlib.pyplot as pyplot
import IPython


import pandas as pd
import seaborn as sns
from pathlib import Path

import script.utils as me

def load_key_bytes(path: str) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    key = np.fromfile(p, dtype=np.uint8)
    if key.size == 0:
        return None
    return key

def save_key_bytes(path: str, key: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(key, dtype=np.uint8).tofile(p)

def get_or_create_key_bytes(
    key_path: str,
    *,
    seed: float = 0.0123,
    r: float = 3.9159,
    big_num: int = 10_000,
    no_cache=False
) -> np.ndarray:
    if(no_cache==False):
        key = load_key_bytes(key_path)
        if key is not None:
            print(f"[key] loaded: {key_path} ({key.size} bytes)")
            return key

    print(f"[key] not found, generating: {key_path}")
    mergedfinal = build_mergedfinal(seed, r, big_num)
    key = (np.array(mergedfinal, dtype=np.uint64) % 256).astype(np.uint8)
    save_key_bytes(key_path, key)
    print(f"[key] saved: {key_path} ({key.size} bytes)")
    return key


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)  

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs


""" plot spectrogram"""
def spectogram(audiopath, binsize=2**10, plotpath=None, colormap="jet"): #<------THIS ONEEEEEEEEEE
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()
    
    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    return ims


def printlst (lst) :
  return
  print('[',lst[0],lst[1],lst[2],lst[3],lst[4],lst[5],'......',len(lst),"items ]")



def keygen(x, r, size):
    key = np.empty(size, dtype=np.float64)
    for i in range(size):
        x = r * x * (1 - x)
        key[i] = ((x * (10**16)) % 256.126)
    return key

import re

def build_mergedfinal(seed, r, big_num):
    # 1) keygen ОДИН раз
    k = keygen(seed, r, big_num)  # float массив

    # 2) deckey: дробная часть (k - int(k))
    deckey = k - np.floor(k)

    # 3) finkey: последние 3 символа строки дроби
    finkey = [int(str(v)[-3:]) for v in deckey]

    # 4) binkey: bin(...)
    binkey = [bin(v) for v in finkey]

    # 5) binkey_fin: regex digits
    binkey_fin = [re.findall(r"\d+", s) for s in binkey]

    # 6) merged: flatten
    merged = list(itertools.chain(*binkey_fin))

    # 7) del merged[0::2]
    del merged[0::2]

    # 8) mergedfinal: str -> int
    mergedfinal = list(map(int, merged))

    return mergedfinal




def test_4_run(org, enc, dec, seed = 0.0123, r=3.9159, no_cache=False, errored_enc=None, org_format=None, new_format=None):

    # org= "test4/artefacts/input.wav" #Path to the audio file
    # enc = "test4/artefacts/encrypted.wav"
    # dec = "test4/artefacts/decrypted.wav"
    big_num = 10_000
    key_file = "test4/artefacts/key_u8.bin"   # куда сохранить

    key = get_or_create_key_bytes(
        key_file,
        seed=seed,
        r=r,
        big_num=big_num,
        no_cache=no_cache
    )
    key_len = key.size
    print("Key length:", key_len)




    print("Extract from input audio file")
    print(org)
    w = wave.open(org, 'rb')
    channels = w.getnchannels()
    framerate = w.getframerate()
    sampwidth = w.getsampwidth()

    writer = wave.open(enc, 'wb')
    writer.setnchannels(channels)
    writer.setsampwidth(sampwidth)
    writer.setframerate(framerate)

    # key = np.array(mergedfinal, dtype=np.uint8)  # ключ как байты 0..255
    # key_len = key.size

    BLOCK = 2400  # фреймы (не байты!)
    key_pos = 0

    while True:
        frames = w.readframes(BLOCK)
        if not frames:
            break

        data = np.frombuffer(frames, dtype=np.uint8)

        # делаем ключ нужной длины под этот блок
        idx = (key_pos + np.arange(data.size)) % key_len
        ks = key[idx]
        key_pos = (key_pos + data.size) % key_len

        enc_bytes = np.bitwise_xor(data, ks).tobytes()
        writer.writeframesraw(enc_bytes)

    writer.close()
    w.close()
    print("Written to file ", enc)



    print("Now we write the values back into a audio file")
    
    if not errored_enc==None:
        me.make_err(enc, errored_enc)
        enc = errored_enc
    if(org_format != None and new_format != None):
        me.emulate_format_convertion(enc, org_format, new_format)
    w = wave.open(enc, 'rb')
    channels = w.getnchannels()
    framerate = w.getframerate()
    sampwidth = w.getsampwidth()

    writer = wave.open(dec, 'wb')
    writer.setnchannels(channels)
    writer.setsampwidth(sampwidth)
    writer.setframerate(framerate)

    # key = np.array(mergedfinal, dtype=np.uint8)
    # key_len = key.size

    BLOCK = 2400
    key_pos = 0

    while True:
        frames = w.readframes(BLOCK)
        if not frames:
            break

        data = np.frombuffer(frames, dtype=np.uint8)

        idx = (key_pos + np.arange(data.size)) % key_len
        ks = key[idx]
        key_pos = (key_pos + data.size) % key_len


        dec_bytes = np.bitwise_xor(data, ks).tobytes()
        writer.writeframesraw(dec_bytes)

    writer.close()
    w.close()


    print("Written to file ", dec)

    # ims = spectogram(path)
    # print(path)
