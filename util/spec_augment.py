#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import random
from typing import Tuple, List

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def is_npy_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".npy"


def load_spectrogram(path: str) -> Tuple[np.ndarray, str]:
    """
    Load a spectrogram from a PNG/JPG (as [freq, time] float32 0..1) or .npy (as float32).
    Returns (spec[H,W], mode) where mode in {"image","npy"} to control saving behavior.
    Assumes mel spectrograms are encoded with frequency along the vertical axis (height).
    """
    if is_image_file(path):
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required to load image files. Install with `pip install Pillow`.")
        img = Image.open(path).convert("L")  # grayscale
        arr = np.asarray(img).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr, "image"
    elif is_npy_file(path):
        arr = np.load(path)
        if arr.ndim == 3:
            # Accept [C, F, T] or [H, W, C]; reduce if possible
            if arr.shape[0] in (1, 2, 3) and arr.shape[-1] not in (1, 2, 3):
                arr = arr[0]
            elif arr.shape[-1] in (1, 2, 3) and arr.shape[0] not in (1, 2, 3):
                arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram in {path}, got shape {arr.shape}")
        arr = arr.astype(np.float32)
        a_min, a_max = float(np.min(arr)), float(np.max(arr))
        if not (0.0 <= a_min and a_max <= 1.0):
            rng = a_max - a_min if a_max != a_min else 1.0
            arr = (arr - a_min) / rng
        return arr, "npy"
    else:
        raise ValueError(f"Unsupported file type: {path}")


def save_spectrogram(path_in: str, arr: np.ndarray, mode: str, out_suffix: str, sample_idx: int, as_uint8: bool) -> str:
    base, ext = os.path.splitext(path_in)
    out_path = f"{base}{out_suffix}{sample_idx}{ext if mode=='image' else '.npy'}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr_clip = np.clip(arr, 0.0, 1.0)
    if mode == "image":
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required to save image files. Install with `pip install Pillow`.")
        if as_uint8:
            out_img = (arr_clip * 255.0 + 0.5).astype(np.uint8)
            img = Image.fromarray(out_img, mode="L")
        else:
            out_img = (arr_clip * 65535.0 + 0.5).astype(np.uint16)
            img = Image.fromarray(out_img, mode="I;16")
        img.save(out_path)
    else:
        np.save(out_path, arr_clip.astype(np.float32))
    return out_path


def apply_specaugment(
    spec: np.ndarray,
    freq_masks: int = 1,
    time_masks: int = 1,
    freq_width: int = 15,
    time_width: int = 40,
    mask_value: float = 0.0,
    freq_width_pct: float = 0.0,
    time_width_pct: float = 0.0,
) -> np.ndarray:
    """
    Apply SpecAugment (frequency & time masking) to a [F, T] spectrogram in-place copy.
    - freq_width/time_width are absolute maxima (bins/frames).
    - freq_width_pct/time_width_pct are optional percentages of F/T (0..1). If > 0, they override absolute widths.
    """
    F, T = spec.shape
    out = spec.copy()

    max_fw = max(1, int(freq_width_pct * F)) if freq_width_pct > 0 else max(1, freq_width)
    max_tw = max(1, int(time_width_pct * T)) if time_width_pct > 0 else max(1, time_width)

    # Frequency masks
    for _ in range(freq_masks):
        fw = random.randint(1, min(max_fw, F))
        f0 = random.randint(0, F - fw)
        out[f0:f0 + fw, :] = mask_value

    # Time masks
    for _ in range(time_masks):
        tw = random.randint(1, min(max_tw, T))
        t0 = random.randint(0, T - tw)
        out[:, t0:t0 + tw] = mask_value

    return out


def find_files(root: str, patterns: List[str], recursive: bool) -> List[str]:
    files = []
    for pat in patterns:
        if recursive:
            files.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
        else:
            files.extend(glob.glob(os.path.join(root, pat), recursive=False))
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply SpecAugment to mel spectrograms (PNG/JPG or .npy).")
    parser.add_argument("--root", type=str, default=".", help="Root directory to scan (default: current directory).")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern for files (e.g., *.png or *.npy).")
    parser.add_argument("--extra-pattern", type=str, default="", help="Optional second glob pattern (e.g., *.npy).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--samples-per-file", type=int, default=1, help="How many augmented samples to generate per file.")
    parser.add_argument("--freq-masks", type=int, default=2, help="Number of frequency masks.")
    parser.add_argument("--time-masks", type=int, default=2, help="Number of time masks.")
    parser.add_argument("--freq-width", type=int, default=12, help="Max width (in mel bins). Ignored if --freq-width-pct > 0.")
    parser.add_argument("--time-width", type=int, default=40, help="Max width (in frames). Ignored if --time-width-pct > 0.")
    parser.add_argument("--freq-width-pct", type=float, default=0.0, help="Max width as fraction of F (0..1). Overrides --freq-width if > 0.")
    parser.add_argument("--time-width-pct", type=float, default=0.0, help="Max width as fraction of T (0..1). Overrides --time-width if > 0.")
    parser.add_argument("--mask-value", type=float, default=0.0, help="Value to fill masked regions with (0..1). Use dataset mean for softer masking.")
    parser.add_argument("--mask-with-mean", action="store_true", help="Fill masked regions with per-image mean instead of --mask-value.")
    parser.add_argument("--out-suffix", type=str, default="_specaug_", help="Suffix added before extension, plus an index.")
    parser.add_argument("--uint8", action="store_true", help="Save images as 8-bit (0..255). If not set, PNGs are saved as 16-bit where possible.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (0 = random).")
    args = parser.parse_args(argv)

    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    patterns = [args.pattern]
    if args.extra_pattern:
        patterns.append(args.extra_pattern)

    paths = find_files(args.root, patterns, args.recursive)
    if not paths:
        print("No files matched. Adjust --root/--pattern/--recursive.", file=sys.stderr)
        return 2

    print(f"Found {len(paths)} file(s). Processing...")

    for path in paths:
        try:
            spec, mode = load_spectrogram(path)
            mask_val = float(np.mean(spec)) if args.mask_with_mean else float(args.mask_value)
            for i in range(args.samples_per_file):
                aug = apply_specaugment(
                    spec,
                    freq_masks=args.freq_masks,
                    time_masks=args.time_masks,
                    freq_width=args.freq_width,
                    time_width=args.time_width,
                    mask_value=mask_val,
                    freq_width_pct=args.freq_width_pct,
                    time_width_pct=args.time_width_pct,
                )
                out_path = save_spectrogram(path, aug, mode, args.out_suffix, i, as_uint8=bool(args.uint8))
                print(f"Wrote: {out_path}")
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
