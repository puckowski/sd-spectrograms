import librosa
import soundfile as sf
import matplotlib.image as mpimg
import numpy as np
import glob
import os
import scipy.signal

SR = 16000
DB_MIN = -80
DB_MAX = 0
N_FFT = 2048
HOP_LENGTH = 512

for path in glob.glob("spectrogram_*.png"):
    S_dB = mpimg.imread(path)
    if S_dB.ndim == 3:
        S_dB = np.mean(S_dB, axis=2)
    # Undo normalization (assume float image 0-1 or uint8 0-255)
    if S_dB.max() > 1.1:
        S_dB = S_dB / 255 * (DB_MAX - DB_MIN) + DB_MIN
    else:
        S_dB = S_dB * (DB_MAX - DB_MIN) + DB_MIN
    # dB to power
    S = librosa.db_to_power(S_dB)
    # Invert mel to audio with more iterations
    y = librosa.feature.inverse.mel_to_audio(
        S, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=256
    )
    # Optional: lowpass filter to reduce hiss
    sos = scipy.signal.butter(10, 7000, 'low', fs=SR, output='sos')
    y = scipy.signal.sosfilt(sos, y)
    # Optional: denoise (pip install noisereduce)
    # import noisereduce as nr
    # y = nr.reduce_noise(y=y, sr=SR)
    wav_path = os.path.splitext(path)[0] + ".wav"
    sf.write(wav_path, y, SR)
    print(f"Wrote {wav_path}")
