# pip install torch pillow tqdm open_clip_torch
import os, glob, csv, math
from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

# ---------------------------
# Config
# ---------------------------
MODEL_NAME   = "ViT-L-14"
PRETRAINED   = "openai"
BATCH_SIZE   = 16     # increase if you have VRAM
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CSV_OUT      = "clip_image_pairwise.csv"

# ---------------------------
# Pair discovery
# ---------------------------
def find_pairs() -> List[Tuple[Path, Path, str]]:
    """
    Return list of (path_a, path_b, stem) where path_a is 'name.png' and path_b is 'name_2.png'.
    Only includes pairs where both exist.
    """
    all_pngs = [Path(p) for p in glob.glob("*.png")]
    stems = {p.stem for p in all_pngs}

    pairs = []
    for p in all_pngs:
        stem = p.stem
        # We consider 'name.png' as A and 'name_2.png' as B
        if stem.endswith("_2"):
            base_stem = stem[:-2]
            a = Path(base_stem + ".png")
            b = p
        else:
            a = p
            b = Path(stem + "_2.png")
        if a.exists() and b.exists():
            # Use the base stem (without _2) as identifier
            base = a.stem
            pairs.append((a.resolve(), b.resolve(), base))
    # Deduplicate by base stem
    seen = set()
    uniq = []
    for a,b,base in pairs:
        if base not in seen:
            uniq.append((a,b,base))
            seen.add(base)
    return sorted(uniq, key=lambda x: x[2].lower())

# ---------------------------
# CLIP setup
# ---------------------------
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()
    return model, preprocess

@torch.no_grad()
def encode_images(model, images: List[Image.Image]) -> torch.Tensor:
    """Encode a list of PIL images to L2-normalized CLIP features."""
    feats = []
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i+BATCH_SIZE]
        ims = [preprocess(im).to(DEVICE, dtype=dtype) for im in batch]
        ims = torch.stack(ims, dim=0)
        # Some open_clip models support encode_image only; no autocast needed beyond dtype
        emb = model.encode_image(ims)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb)
    return torch.cat(feats, dim=0) if feats else torch.empty(0, model.visual.output_dim)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    pairs = find_pairs()
    if not pairs:
        print("No pairs found. Expected files like 'name.png' and 'name_2.png' in the current folder.")
        raise SystemExit(0)

    print(f"Found {len(pairs)} pairs.")
    model, preprocess = load_model()

    # Load images
    imgs_a, imgs_b, ids = [], [], []
    for a,b,base in pairs:
        try:
            im_a = Image.open(a).convert("RGB")
            im_b = Image.open(b).convert("RGB")
        except Exception as e:
            print(f"[skip] {base}: failed to open -> {e}")
            continue
        imgs_a.append(im_a)
        imgs_b.append(im_b)
        ids.append(base)

    if not ids:
        print("No valid images to compare.")
        raise SystemExit(0)

    # Encode
    feats_a = encode_images(model, imgs_a)
    feats_b = encode_images(model, imgs_b)

    # Cosine similarity (since features are L2-normalized, dot = cosine)
    sims = (feats_a * feats_b).sum(dim=-1).clamp(-1, 1).tolist()
    dists = [1.0 - s for s in sims]

    # Summary stats
    def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
    def median(xs):
        if not xs: return float("nan")
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        mid = n // 2
        if n % 2 == 1:
            return xs_sorted[mid]
        return 0.5 * (xs_sorted[mid-1] + xs_sorted[mid])

    avg_sim, med_sim = mean(sims), median(sims)
    avg_dist, med_dist = mean(dists), median(dists)

    # Print per-file and summary
    print("\nPer-pair results (cosine similarity, distance = 1 − similarity):")
    for base, s, d in zip(ids, sims, dists):
        print(f"{base:40s}  sim={s:.4f}  dist={d:.4f}")

    print("\nSummary:")
    print(f"Mean  similarity: {avg_sim:.4f}")
    print(f"Median similarity: {med_sim:.4f}")
    print(f"Mean  distance  : {avg_dist:.4f}")
    print(f"Median distance : {med_dist:.4f}")

    # Save CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "file_a", "file_b", "cosine_similarity", "distance_1_minus_sim"])
        for (a,b,base), s, d in zip(pairs, sims, dists):
            w.writerow([base, str(a), str(b), f"{s:.6f}", f"{d:.6f}"])
    print(f"\nWrote {CSV_OUT}")
