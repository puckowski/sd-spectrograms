import os
import re
import random

spectrogram_dir = "reg_spectrograms/20_song"

for f in os.listdir(spectrogram_dir):
    if f.lower().endswith((".png", ".jpg")):
        text = 'electronic spectrogram'
        # Write to txt file
        txt_file = os.path.splitext(f)[0] + ".txt"
        with open(os.path.join(spectrogram_dir, txt_file), "w", encoding="utf-8") as fp:
            fp.write(text)
