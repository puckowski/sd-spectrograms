import os, csv, glob, librosa, numpy as np, pyloudnorm as pyln, matplotlib.pyplot as plt
import soundfile as sf

# ---------- Settings ----------
CSV_FILE       = "../metadata.csv"        # <-- your CSV file
AUDIO_ROOT     = "../audio"               # folder that contains subfolders like 34/1004034.mp3
OUTPUT_FOLDER  = "../spectrograms/20_song"  # images + captions saved here (LoRA expects same folder)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SR         = 16000
N_MELS     = 256
DURATION   = 15
OFFSET     = 15
DB_MIN, DB_MAX = -80, 0

TARGET_LUFS    = -14.0    # try -16 for more headroom
TP_LIMIT_DB    = -1.0     # post-normalization peak limit
OVERSAMPLE     = 2        # inter-sample peak check (2–4)

# ---------- Helpers ----------
def pad_to_multiple_of_8(x, axis=1):
    length = x.shape[axis]
    pad_len = (8 - (length % 8)) % 8
    if pad_len == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return np.pad(x, pad_width, mode='constant')

def lufs_normalize_with_limiting(y, sr, target_lufs=-14.0, tp_limit_db=-1.0, oversample=2):
    meter = pyln.Meter(sr)                      # ITU-R BS.1770 integrated loudness
    loudness = meter.integrated_loudness(y)

    gain_db = target_lufs - loudness
    y_norm = y * (10.0 ** (gain_db / 20.0))

    if oversample > 1:
        y_up = np.interp(
            np.arange(0, len(y_norm), 1.0/oversample),
            np.arange(len(y_norm)),
            y_norm
        )
        peak = np.max(np.abs(y_up))
    else:
        peak = np.max(np.abs(y_norm))

    limit_lin = 10.0 ** (tp_limit_db / 20.0)
    if peak > limit_lin:
        y_norm = y_norm * (limit_lin / (peak + 1e-12))

    return y_norm, loudness, gain_db

def safe_bool(val):
    # Accept "True"/"False" (case-insensitive) or real booleans
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"

# ---------- Main ----------
with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, start=1):
        try:
            if not safe_bool(row.get("is_valid_subset", True)):
                continue

            rel_path = row["path"].strip()  # e.g., 34/1004034.mp3

            # insert ".2min" before ".mp3"
            if rel_path.lower().endswith(".mp3"):
                rel_path = rel_path[:-4] + ".2min.mp3"

            audio_path = os.path.join(AUDIO_ROOT, rel_path)
            if not os.path.isfile(audio_path):
                print(f"[skip missing] {audio_path}")
                continue

            # Choose a stem that pairs files (png + txt). Use basename without extension.
            # --- Build a unique, safe stem from the CSV path (folder + filename, no ext) ---
            # Use the ORIGINAL CSV path for naming (so we don't bake ".2min" into names).
            csv_rel = (row["path"] or "").strip().replace("\\", "/")   # e.g., "34/1004034.mp3"
            stem_no_ext = os.path.splitext(csv_rel)[0]                  # -> "34/1004034"

            # Make filename-safe: replace "/" and any odd chars with "_"
            safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem_no_ext)
            safe_stem = safe_stem.replace("/", "_")                     # -> "34_1004034"

            png_out = os.path.join(OUTPUT_FOLDER, f"{safe_stem}.png")
            txt_out = os.path.join(OUTPUT_FOLDER, f"{safe_stem}.txt")
            wav_out = os.path.join(OUTPUT_FOLDER, f"{safe_stem}_norm.wav")  # optional

            # Load audio window
            y, sr = librosa.load(audio_path, sr=SR, offset=OFFSET, duration=DURATION)

            # Mild pre-emphasis (helps reduce low-end dominance)
            y = librosa.effects.preemphasis(y)

            # Loudness normalize + limit
            y_norm, in_lufs, applied_gain_db = lufs_normalize_with_limiting(
                y, SR, target_lufs=TARGET_LUFS, tp_limit_db=TP_LIMIT_DB, oversample=OVERSAMPLE
            )

            # Mel spectrogram
            S = librosa.feature.melspectrogram(y=y_norm, sr=SR, n_mels=N_MELS, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            S_norm = np.clip((S_db - DB_MIN) / (DB_MAX - DB_MIN), 0, 1)
            S_norm = pad_to_multiple_of_8(S_norm, axis=1)

            # Save spectrogram (grayscale 0–1)
            plt.imsave(png_out, S_norm, cmap='gray')

            # (Optional) save normalized window for reference
            # sf.write(wav_out, y_norm, SR)

            # Save caption next to image (LoRA pairing)
            caption = (row.get("caption") or "").strip()
            with open(txt_out, "w", encoding="utf-8") as tf:
                tf.write(caption + "\n")

            print(f"[ok] {safe_stem} | LUFS_in={in_lufs:.2f} | gain_db={applied_gain_db:.2f} -> {png_out}")

        except Exception as e:
            print(f"[error row {i}] {e}")
