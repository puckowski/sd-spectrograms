import matplotlib.image as mpimg
import librosa
import soundfile as sf
import numpy as np
import glob
import os

SR = 16000
N_MELS = 256
DURATION = 15
DB_MIN = -80
DB_MAX = 0
output_audio_folder = "reconstructed_audio"
os.makedirs(output_audio_folder, exist_ok=True)

for path in glob.glob("spectrograms/20_song/*.png"):
    # Load normalized image
    S_img = mpimg.imread(path)
    # If the image has 3rd (color) channel, drop it (use grayscale)
    if S_img.ndim == 3:
        S_img = S_img[..., 0]

    # If the shape is (472, 256), transpose to (256, 469)
    if S_img.shape == (472, 256):
        S_img = S_img.T
    elif S_img.shape != (256, 472):
        print(f"Skipping {path}: Unexpected shape {S_img.shape}")
        continue

    # Rescale from [0, 1] back to [DB_MIN, DB_MAX]
    S_db = S_img * (DB_MAX - DB_MIN) + DB_MIN

    # Convert dB to power
    S = librosa.db_to_power(S_db)

    # Invert mel to linear spectrogram
    y = librosa.feature.inverse.mel_to_audio(
        S,
        sr=SR,
        n_fft=2048,
        hop_length=512,
        n_iter=32,
        win_length=2048,
        power=2.0
    )
    # Ensure 15 seconds
    y = y[:SR * DURATION]

    fname = os.path.splitext(os.path.basename(path))[0]
    sf.write(f"{output_audio_folder}/{fname}_reconstructed.wav", y, SR)
    print(f"Wrote: {output_audio_folder}/{fname}_reconstructed.wav")
