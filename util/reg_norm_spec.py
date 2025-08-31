import librosa, numpy as np, pyloudnorm as pyln, matplotlib.pyplot as plt
import soundfile as sf, glob, os

SR = 16000
N_MELS = 256
DURATION = 15
DB_MIN, DB_MAX = -80, 0

TARGET_LUFS = -14.0     # streaming-friendly; try -16 for more headroom
TP_LIMIT_DB = -1.0      # peak limit after normalization (dBFS)
OVERSAMPLE = 2          # 2–4 helps catch inter-sample peaks

audio_folder = "music"
output_folder = "reg_spectrograms/20_song"
os.makedirs(output_folder, exist_ok=True)

def pad_to_multiple_of_8(x, axis=1):
    length = x.shape[axis]
    pad_len = (8 - (length % 8)) % 8
    if pad_len == 0: return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return np.pad(x, pad_width, mode='constant')

def lufs_normalize_with_limiting(y, sr, target_lufs=-14.0, tp_limit_db=-1.0, oversample=2):
    # 1) Measure integrated loudness
    meter = pyln.Meter(sr)  # ITU-R BS.1770
    loudness = meter.integrated_loudness(y)

    # 2) Normalize to target LUFS
    gain_db = target_lufs - loudness
    y_norm = y * (10.0 ** (gain_db / 20.0))

    # 3) Simple true-peak-ish limiting: detect peak at higher rate then scale if needed
    if oversample > 1:
        # cheap oversampling via linear interpolation
        y_up = np.interp(
            np.arange(0, len(y_norm), 1.0/oversample),
            np.arange(len(y_norm)),
            y_norm
        )
        peak = np.max(np.abs(y_up))
    else:
        peak = np.max(np.abs(y_norm))

    peak_dbfs = 20.0 * np.log10(max(peak, 1e-12))
    limit_lin = 10.0 ** (tp_limit_db / 20.0)  # e.g., -1 dBFS -> 0.89125
    if peak > limit_lin:
        y_norm = y_norm * (limit_lin / (peak + 1e-12))

    return y_norm, loudness, gain_db

for path in glob.glob(f"{audio_folder}/*.*"):
    # Load 15 seconds starting at 15s
    y, sr = librosa.load(path, sr=SR, offset=15, duration=DURATION)

    # Optional: light noise taming BEFORE normalization (high-pass at 40 Hz)
    y = librosa.effects.preemphasis(y)  # mild HF tilt; keeps low-end from dominating

    # Loudness-normalize and limit
    y_norm, in_lufs, applied_gain_db = lufs_normalize_with_limiting(
        y, SR, target_lufs=TARGET_LUFS, tp_limit_db=TP_LIMIT_DB, oversample=OVERSAMPLE
    )

    # Mel spectrogram from normalized audio
    S = librosa.feature.melspectrogram(y=y_norm, sr=SR, n_mels=N_MELS, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = np.clip((S_db - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
    S_norm = pad_to_multiple_of_8(S_norm, axis=1)

    # Save PNG
    fname = os.path.splitext(os.path.basename(path))[0]
    plt.imsave(f"{output_folder}/{fname}.png", S_norm, cmap='gray')

    # (Optional) Also save the normalized audio so you have the louder version
    # sf.write(f"{output_folder}/{fname}_norm.wav", y_norm, SR)
