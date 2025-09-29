#!/usr/bin/env python3
"""
Read `metadata.csv`, take each `path` like `87/103887.mp3` from the `audio/` folder,
and export a 15-second clip starting at t=15s to `reconstructed_audio/` as
`<filename_without_ext>_audio.wav` (e.g., 103887_audio.wav).

Requires: pydub and FFmpeg (ffmpeg/ffprobe available on PATH).
    pip install pydub
    # Install FFmpeg per your OS and ensure `ffmpeg` is on PATH
"""

import csv
import sys
from pathlib import Path
from pydub import AudioSegment

# --- Config (tweak if you want) ---
METADATA_CSV = Path("../metadata.csv")
AUDIO_ROOT   = Path("../audio")
OUT_ROOT     = Path("../reconstructed_audio")
START_MS     = 15_000           # start at 15 seconds
DURATION_MS  = 15_000           # 15-second clip (ends at 30s)
OUTPUT_FMT   = "wav"            # export format (wav to avoid re-encoding loss)
# ----------------------------------

def main():
    if not METADATA_CSV.exists():
        print(f"ERROR: {METADATA_CSV} not found", file=sys.stderr)
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Read CSV and ensure 'path' column exists
    with METADATA_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames:
            print("ERROR: metadata.csv must have a 'path' column.", file=sys.stderr)
            sys.exit(1)

        total = 0
        made  = 0
        skipped_short = 0
        missing = 0
        failed = 0

        for row in reader:
            total += 1
            rel_path = (row.get("path") or "").strip()
            if not rel_path:
                continue

            # Insert ".2min" before ".mp3" (only if it ends with .mp3)
            if rel_path.lower().endswith(".mp3"):
                rel_path = rel_path[:-4] + ".2min.mp3"

            in_path = AUDIO_ROOT / rel_path
            if not in_path.exists():
                missing += 1
                print(f"WARNING: Missing audio file: {in_path}", file=sys.stderr)
                continue

            # Output file name is based on the filename of the path + "_audio"
            # e.g., "87/103887.mp3" -> "103887_audio.wav"
            base_name = in_path.stem  # "103887"
            out_path  = OUT_ROOT / f"{base_name}_audio.{OUTPUT_FMT}"

            try:
                audio = AudioSegment.from_file(in_path)
                if len(audio) < START_MS + DURATION_MS:
                    # Not enough audio to extract 15s starting at 15s
                    skipped_short += 1
                    print(f"NOTE: Too short (< {START_MS + DURATION_MS} ms): {in_path}", file=sys.stderr)
                    continue

                clip = audio[START_MS: START_MS + DURATION_MS]
                # Export as 16-bit PCM WAV by default
                clip.export(out_path, format=OUTPUT_FMT)
                made += 1
                print(f"OK: {out_path}")

            except Exception as e:
                failed += 1
                print(f"ERROR processing {in_path}: {e}", file=sys.stderr)

    print("\n--- Summary ---")
    print(f"Rows read:          {total}")
    print(f"Clips created:      {made}")
    print(f"Missing files:      {missing}")
    print(f"Skipped (too short):{skipped_short}")
    print(f"Failed:             {failed}")

if __name__ == "__main__":
    main()
