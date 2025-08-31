import librosa
import numpy as np
import cv2
import soundfile as sf
import os

# Parameters from your original process
n_mels = 256
sr = 16000
img_folder = "spectrograms/20_song"
output_folder = "reconstructed_audio"
os.makedirs(output_folder, exist_ok=True)

# Your original dB range for display; adjust if different
db_min, db_max = -80, 0

for img_file in os.listdir(img_folder):
    if not img_file.endswith('.png'):
        continue
    # Load image as grayscale, shape: (n_mels, time)
    img_path = os.path.join(img_folder, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Convert to float32, normalize 0-1
    S_dB_norm = img.astype(np.float32) / 255.0
    # Map back to dB scale
    S_dB = S_dB_norm * (db_max - db_min) + db_min
    # Undo dB to power
    S = librosa.db_to_power(S_dB, ref=1.0)
    # Invert Mel to audio
    y = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_iter=64, n_fft=2048, hop_length=512)
    # Save as WAV
    base = os.path.splitext(img_file)[0]
    out_wav = os.path.join(output_folder, f"{base}_recon.wav")
    sf.write(out_wav, y, sr)
    print(f"Saved: {out_wav}")