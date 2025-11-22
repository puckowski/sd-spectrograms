import os
import sys
import soundfile as sf

def clip_mp3s(directory: str, start_sec=15, end_sec=30):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    output_dir = os.path.join(directory, "clipped")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.lower().endswith(".mp3"):
            input_path = os.path.join(directory, filename)
            try:
                # Open MP3 file
                with sf.SoundFile(input_path, "r") as f:
                    sr = f.samplerate
                    start_frame = int(start_sec * sr)
                    end_frame = int(end_sec * sr)

                    # Seek to start
                    f.seek(start_frame)
                    # Read only desired frames
                    frames = f.read(frames=end_frame - start_frame)

                # Save clipped portion
                output_path = os.path.join(output_dir, f"clipped_{filename}")
                sf.write(output_path, frames, sr, format="MP3")
                print(f"Clipped {filename} -> {output_path}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clip_mp3s.py <directory>")
    else:
        clip_mp3s(sys.argv[1])
