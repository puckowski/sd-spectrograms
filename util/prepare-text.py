import os
import re
import random

spectrogram_dir = "spectrograms/20_song"

rock_descriptors = [
    "Energetic", "Gritty", "Anthemic", "Distorted", "Melodic", "Powerful", "Rebellious",
    "Intense", "Catchy", "Rhythmic", "Dynamic", "Raw", "Aggressive", "Electric",
    "Loud", "Passionate", "Driving", "Heavy", "Edgy", "Upbeat"
]

for f in os.listdir(spectrogram_dir):
    if f.lower().endswith((".png", ".jpg")):
        name = os.path.splitext(f)[0]
        # Extract alphabetic words from filename
        words = re.findall(r'\b[a-zA-Z]+\b', name)
        # Pick 3 random descriptors
        random_words = random.sample(rock_descriptors, 2)
        # Combine and join
        all_words = words + random_words
        text = ' '.join(all_words)
        text += ' spectrogram'
        # Write to txt file
        txt_file = os.path.splitext(f)[0] + ".txt"
        with open(os.path.join(spectrogram_dir, txt_file), "w", encoding="utf-8") as fp:
            fp.write(text)
