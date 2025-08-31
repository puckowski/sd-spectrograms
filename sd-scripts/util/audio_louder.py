from pydub import AudioSegment
from pydub.effects import low_pass_filter
import glob
import os

folder = "reconstructed_audio"   # Folder with .wav files
gain_db = 27                    # Volume increase (dB)
cutoff_freq = 7000              # Adjust cutoff to taste (Hz)

for filepath in glob.glob(os.path.join(folder, "*.wav")):
    dirname, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(dirname, f"{name}_louder{ext}")

    audio = AudioSegment.from_wav(filepath)

    # --- Filter before amplifying ---
    filtered_audio = low_pass_filter(audio, cutoff_freq)
    louder_audio = filtered_audio + gain_db

    louder_audio.export(output_path, format="wav")
    print(f"Processed {filename} -> {output_path}")

print("All done!")
