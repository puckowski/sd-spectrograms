# inverse_mel_images_to_audio.py
# Requires: pip install librosa soundfile numpy matplotlib pyloudnorm pillow tqdm
import os, glob, warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image
from tqdm import tqdm

import librosa
import pyloudnorm as pyln

# ---------- Settings (match your forward pass) ----------
IMAGES_FOLDER   = "./"    # where *.png live
OUTPUT_FOLDER   = "reconstructed_audio"     # where *.wav will be written
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SR         = 16000
N_MELS     = 256
FMAX       = 8000
# librosa.melspectrogram defaults we need to mirror
N_FFT      = 2048
HOP        = 512
WIN_LENGTH = None      # librosa default = None -> equals n_fft internally for STFT
POWER      = 2.0       # power spectrogram (not magnitude)

# Your forward normalization window ->
DB_MIN, DB_MAX = -80.0, 0.0   # (S_db - DB_MIN)/(DB_MAX-DB_MIN)

# Pre/De-emphasis
PREEMPH_COEF = 0.97           # default used by librosa.effects.preemphasis
APPLY_DEEMPH = True           # True to invert your pre-emphasis step

# Loudness normalization / limiting (optional but recommended)
APPLY_LUFS_NORM = True
TARGET_LUFS     = -14.0
TP_LIMIT_DB     = -1.0        # peak limiter after LUFS normalization
OVERSAMPLE      = 2           # inter-sample peak check (2–4)

# Griffin-Lim iterations
GL_ITERS = 64

# If you know your target duration (in seconds) per image, set it to trim/pad the output
# Otherwise leave as None and we won't force a length.
FORCE_DURATION_SEC = 15.0

# ---------- Helpers ----------
def read_grayscale_01(path):
    """Read an image and return float32 grayscale in [0,1] with shape [H, W]."""
    img = Image.open(path).convert("L")      # force 8-bit gray
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def denorm_db(S_norm, db_min=-80.0, db_max=0.0):
    """Invert (S_db - db_min)/(db_max - db_min) → S_db."""
    return S_norm * (db_max - db_min) + db_min

def lufs_normalize_with_limiting(y, sr, target_lufs=-14.0, tp_limit_db=-1.0, oversample=2):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    gain_db = target_lufs - loudness
    y_norm = y * (10.0 ** (gain_db / 20.0))

    # True-peak-ish check via oversampling
    if oversample > 1:
        # linear interpolation oversampling is OK for a quick TP check
        x = np.arange(len(y_norm), dtype=np.float32)
        xi = np.linspace(0, len(y_norm) - 1, len(y_norm) * oversample, dtype=np.float32)
        y_up = np.interp(xi, x, y_norm)
        peak = np.max(np.abs(y_up))
    else:
        peak = np.max(np.abs(y_norm))

    limit_lin = 10.0 ** (tp_limit_db / 20.0)
    if peak > limit_lin:
        y_norm = y_norm * (limit_lin / (peak + 1e-12))

    return y_norm, loudness, gain_db

def maybe_deemphasis(y, coef=0.97):
    # librosa has effects.deemphasis; do a small fallback if not available
    try:
        return librosa.effects.deemphasis(y, coef=coef)
    except Exception:
        # IIR inverse of preemphasis: y[n] = x[n] + a*y[n-1] in a rough sense.
        # Simple FIR-ish approximation:
        z = np.copy(y)
        for n in range(1, len(z)):
            z[n] = y[n] + coef * z[n - 1]
        return z

def frames_to_length(num_frames, hop_length, win_length, center=True):
    """Approximate audio length (samples) from number of STFT frames."""
    # librosa’s default padding when center=True adds win_length//2 at both ends.
    # mel_to_audio can accept 'length' to trim; we’ll compute a reasonable target length.
    if win_length is None:
        win_length = N_FFT
    pad = win_length // 2 if center else 0
    return max(0, num_frames * hop_length)

# ---------- Main ----------
def reconstruct_one(png_path):
    # 1) Read normalized grayscale mel image [N_MELS, T] in [0,1]
    S_norm = read_grayscale_01(png_path)

    # Sanity: ensure it’s [n_mels, time]
    # Your forward code wrote plt.imsave on S_norm without transpose, so height=mel bins, width=time.
    if S_norm.shape[0] != N_MELS:
        # If someone saved transposed, recover automatically
        if S_norm.shape[1] == N_MELS:
            S_norm = S_norm.T
        else:
            raise ValueError(f"{png_path}: Expected one dimension = {N_MELS}, got {S_norm.shape}")

    # 2) De-normalize back to dB, then to power-mel
    S_db = denorm_db(S_norm, DB_MIN, DB_MAX).astype(np.float32)
    S_mel_power = librosa.db_to_power(S_db)  # POWER scale

    # 3) Invert mel → audio with Griffin-Lim
    #    We can pass 'length' to help avoid long tails from zero-padded columns
    T = S_mel_power.shape[1]

    # librosa.feature.inverse.mel_to_audio handles filterbank inversion + Griffin-Lim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = librosa.feature.inverse.mel_to_audio(
            M=S_mel_power,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=WIN_LENGTH,
            power=POWER,
            n_iter=GL_ITERS,
            fmin=0.0,
            fmax=FMAX,
            center=True
        )

    # 4) Apply de-emphasis to invert your pre-emphasis (optional but recommended)
    if APPLY_DEEMPH:
        y = maybe_deemphasis(y, coef=PREEMPH_COEF)

    # 5) LUFS normalize + limit (optional; you normalized on the way in)
    loudness_in = None
    applied_gain_db = 0.0
    if APPLY_LUFS_NORM:
        y, loudness_in, applied_gain_db = lufs_normalize_with_limiting(
            y, SR, target_lufs=TARGET_LUFS, tp_limit_db=TP_LIMIT_DB, oversample=OVERSAMPLE
        )

    # 6) Final tidy up: hard trim/pad to exact duration if requested
    if FORCE_DURATION_SEC is not None:
        need = int(round(FORCE_DURATION_SEC * SR))
        if len(y) > need:
            y = y[:need]
        elif len(y) < need:
            y = np.pad(y, (0, need - len(y)), mode="constant")

    return y, loudness_in, applied_gain_db

def main():
    pngs = sorted(glob.glob(os.path.join(IMAGES_FOLDER, "*.png")))
    if not pngs:
        print(f"No PNGs found in {IMAGES_FOLDER}")
        return

    for p in tqdm(pngs, desc="Reconstructing"):
        try:
            y, lufs_in, gain_db = reconstruct_one(p)
            stem = Path(p).stem
            wav_out = os.path.join(OUTPUT_FOLDER, f"{stem}_recon.wav")
            sf.write(wav_out, y, SR)
            if lufs_in is not None:
                print(f"[ok] {stem} | LUFS_in≈{lufs_in:.2f} | gain_db={gain_db:.2f} -> {wav_out}")
            else:
                print(f"[ok] {stem} -> {wav_out}")
        except Exception as e:
            print(f"[error] {p}: {e}")

if __name__ == "__main__":
    main()
