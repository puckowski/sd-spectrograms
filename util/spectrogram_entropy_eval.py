#!/usr/bin/env python3
"""
Compute spectral entropy for all spectrogram images in a directory, including:
- Overall spectral entropy
- 4 horizontal section entropies (top->bottom)
- 2 vertical section entropies (left half, right half)

Requires: numpy, pillow, scipy
    pip install numpy pillow scipy

Usage:
    python spectral_entropy_images.py --dir ./spectrograms --patterns "*.png" "*.jpg" --recursive --out results.csv
"""
import argparse
import os
import sys
import glob
import csv
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy.stats import entropy


def spectral_entropy_per_frame(image_2d: np.ndarray) -> float:
    """
    image_2d: [H, W] spectrogram (e.g., [freq, time])
    Returns mean spectral entropy over all time frames.

    We treat axis=0 as frequency bins and axis=1 as time frames.
    """
    if image_2d.ndim != 2:
        raise ValueError(f"Expected 2D array [H, W], got shape {image_2d.shape}")

    # In case negative values appear (e.g., VAE decoded), use magnitude
    S = np.abs(image_2d).astype(np.float64)
    S = S + 1e-8  # Avoid log(0)

    # Normalize per time frame so sum over frequencies = 1
    denom = S.sum(axis=0, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    P = S / denom  # shape [freq, time]

    # Entropy over frequency axis for each time frame
    H = entropy(P, base=2, axis=0)  # shape [time]
    return float(np.mean(H))


def section_entropies(image_2d: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Returns:
      horizontal_entropies: list of 4 entropies for top->bottom sections
      vertical_entropies:   list of 2 entropies for left->right sections
    Uses np.array_split to handle sizes not exactly divisible.
    """
    # 4 horizontal bands (split along H / axis=0)
    h_sections = np.array_split(image_2d, 4, axis=0)
    horizontal_entropies = [spectral_entropy_per_frame(sec) for sec in h_sections]

    # 2 vertical bands (split along W / axis=1)
    v_sections = np.array_split(image_2d, 2, axis=1)
    vertical_entropies = [spectral_entropy_per_frame(sec) for sec in v_sections]

    return horizontal_entropies, vertical_entropies


def load_grayscale_image(path: str) -> np.ndarray:
    """
    Load image as float64 grayscale in range [0,1].
    If the input is RGB, it gets converted to luminance using PIL's 'L' mode.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.float64) / 255.0  # [H, W] in [0,1]
        return arr


def collect_files(directory: str, patterns: List[str], recursive: bool) -> List[str]:
    files = []
    for pat in patterns:
        pattern = os.path.join(directory, "**", pat) if recursive else os.path.join(directory, pat)
        files.extend(glob.glob(pattern, recursive=recursive))
    files = sorted(set(files))
    return files


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Compute spectral entropy for spectrogram images in a directory.")
    parser.add_argument("--dir", required=True, help="Directory containing images")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"],
        help="Glob patterns to match files (space-separated)",
    )
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    parser.add_argument("--out", default="spectral_entropy_results.csv", help="Output CSV path")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any file fails to process")

    args = parser.parse_args(argv)

    files = collect_files(args.dir, args.patterns, args.recursive)
    if not files:
        print("No matching image files found.", file=sys.stderr)
        return 1

    # CSV rows: file, overall, H1..H4, V1..V2, height, width
    results: List[Tuple[str, float, float, float, float, float, float, float, int, int]] = []
    failures: List[Tuple[str, str]] = []

    for i, f in enumerate(files, 1):
        try:
            arr = load_grayscale_image(f)  # [H, W]
            overall = spectral_entropy_per_frame(arr)
            h_vals, v_vals = section_entropies(arr)

            results.append((
                f, overall,
                h_vals[0], h_vals[1], h_vals[2], h_vals[3],
                v_vals[0], v_vals[1],
                arr.shape[0], arr.shape[1]
            ))

            print(
                f"[{i}/{len(files)}] {f}\n"
                f"  overall={overall:.5f} bits  (H={arr.shape[0]}, W={arr.shape[1]})\n"
                f"  horizontal sections (top->bottom): "
                f"[H1={h_vals[0]:.5f}, H2={h_vals[1]:.5f}, H3={h_vals[2]:.5f}, H4={h_vals[3]:.5f}]\n"
                f"  vertical sections (left->right):   "
                f"[V1={v_vals[0]:.5f}, V2={v_vals[1]:.5f}]"
            )

        except Exception as e:
            failures.append((f, str(e)))
            print(f"[{i}/{len(files)}] ERROR {f}: {e}", file=sys.stderr)

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "file",
            "spectral_entropy_mean_bits_overall",
            "spectral_entropy_h1_top",
            "spectral_entropy_h2",
            "spectral_entropy_h3",
            "spectral_entropy_h4_bottom",
            "spectral_entropy_v1_left",
            "spectral_entropy_v2_right",
            "height",
            "width",
        ])
        writer.writerows(results)

    # Summary
    if results:
        per_image_scores = [row[1] for row in results]
        batch_avg = float(np.mean(per_image_scores))
        print(f"\nProcessed {len(results)} files.")
        print(f"Batch average spectral entropy (mean over images): {batch_avg:.5f} bits")
        print(f"CSV written to: {os.path.abspath(args.out)}")
    else:
        print("\nNo successful results.")

    if failures:
        print(f"\n{len(failures)} files failed to process:", file=sys.stderr)
        for f, msg in failures:
            print(f"  {f} -> {msg}", file=sys.stderr)
        if args.fail-on-error:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
