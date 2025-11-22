import os
import soundfile as sf

ROOT_DIR = r"."

START_SEC = 15
END_SEC = 30

def trim_to_wav(in_path):
    print(f"Processing: {in_path}")

    data, samplerate = sf.read(in_path)  # reads MP3 if mpg123 installed

    start = int(START_SEC * samplerate)
    end = int(END_SEC * samplerate)

    if end > len(data):
        print(f"⚠️ Skipped (too short): {in_path}")
        return

    trimmed = data[start:end]

    out_path = os.path.splitext(in_path)[0] + ".clipped.wav"
    sf.write(out_path, trimmed, samplerate)
    print(f"✔ Saved trimmed WAV: {out_path}")


def scan_and_process():
    for root, _, files in os.walk(ROOT_DIR):
        for f in files:
            if f.lower().endswith(".mp3") and "2min" in f.lower():
                trim_to_wav(os.path.join(root, f))


if __name__ == "__main__":
    scan_and_process()
