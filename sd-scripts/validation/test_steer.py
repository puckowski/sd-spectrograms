# pip install torch pillow open_clip_torch lpips torchvision tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch, os, math, csv
from PIL import Image
from tqdm import tqdm
import open_clip
import torchvision.transforms as T
import lpips

# -----------------------
# Config
# -----------------------
MODEL_PATH = "../tools/output_merged_sd"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

HEIGHT, WIDTH = 256, 472
NEGATIVE = "stripes, striped, monotone"

PROMPT = ("Sad instrumental piano song that feels like is telling a story and that one can be put on "
          "in the background to listen while alone mel spectrogram")

SEEDS = [1000 + i for i in range(10)]
GUIDANCE_SCALES = [0.0, 2.0, 4.0, 8.0, 12.0]   # includes a *true control* at 0.0 (unconditional)
STEPS = 32

OUTDIR = "steering_test_out"
CSV_OUT = "steering_results.csv"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------
# Load pipeline
# -----------------------
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,
).to(DEVICE)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
)
if hasattr(pipe.scheduler, "solver_order"):
    pipe.scheduler.solver_order = 2
if hasattr(pipe.scheduler, "lower_order_final"):
    pipe.scheduler.lower_order_final = True

# Optional memory/perf
try: pipe.enable_attention_slicing("max")
except: pass
try: pipe.enable_vae_tiling()
except: pass

# -----------------------
# CLIP + LPIPS setup
# -----------------------
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
clip_model = clip_model.to(DEVICE).eval()
tok = open_clip.get_tokenizer("ViT-L-14")

lpips_loss = lpips.LPIPS(net="vgg").to(DEVICE).eval()
to_tensor = T.ToTensor()

@torch.no_grad()
def clip_text_image_sim(prompt: str, img: Image.Image) -> float:
    im = clip_preprocess(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    tx = tok([prompt]).to(DEVICE)
    im_f = clip_model.encode_image(im); im_f = im_f / im_f.norm(dim=-1, keepdim=True)
    tx_f = clip_model.encode_text(tx); tx_f = tx_f / tx_f.norm(dim=-1, keepdim=True)
    return float((im_f * tx_f).sum())

@torch.no_grad()
def lpips_dist(a: Image.Image, b: Image.Image) -> float:
    a_t = to_tensor(a.convert("RGB")).unsqueeze(0).to(DEVICE)
    b_t = to_tensor(b.convert("RGB")).unsqueeze(0).to(DEVICE)
    return float(lpips_loss(a_t, b_t).mean())

# -----------------------
# Run experiments
# -----------------------
rows = []
for gs in GUIDANCE_SCALES:
    print(f"\n=== guidance_scale = {gs} ===")
    # For each seed, produce an unconditional baseline (gs=0) and a prompted image (this gs)
    for seed in tqdm(SEEDS):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)

        # A) unconditional baseline: guidance_scale = 0 ensures no prompt steering
        img_uncond = pipe(
            prompt="", negative_prompt=NEGATIVE,
            num_inference_steps=STEPS, guidance_scale=0.0,
            height=HEIGHT, width=WIDTH, generator=gen
        ).images[0]

        # B) prompted image: use the same seed but with the tested CFG
        gen2 = torch.Generator(device=DEVICE).manual_seed(seed)
        img_prompted = pipe(
            prompt=PROMPT, negative_prompt=NEGATIVE,
            num_inference_steps=STEPS, guidance_scale=gs,
            height=HEIGHT, width=WIDTH, generator=gen2
        ).images[0]

        # Metrics
        ti_uncond  = clip_text_image_sim(PROMPT, img_uncond)
        ti_prompt  = clip_text_image_sim(PROMPT, img_prompted)
        lp         = lpips_dist(img_uncond, img_prompted)

        # Save a few exemplars for visual inspection
        base = f"seed{seed}_gs{gs:.1f}"
        img_uncond.save(os.path.join(OUTDIR, f"{base}_A_uncond.png"))
        img_prompted.save(os.path.join(OUTDIR, f"{base}_B_prompted.png"))

        rows.append({
            "guidance_scale": gs,
            "seed": seed,
            "clip_text_image_uncond": ti_uncond,
            "clip_text_image_prompted": ti_prompt,
            "clip_delta_prompt_minus_uncond": ti_prompt - ti_uncond,
            "lpips_prompt_vs_uncond": lp,
        })

# -----------------------
# Summaries
# -----------------------
def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

# per-gs aggregation
summary = {}
for gs in GUIDANCE_SCALES:
    subset = [r for r in rows if r["guidance_scale"] == gs]
    summary[gs] = {
        "mean_clip_T·I_uncond": mean([r["clip_text_image_uncond"] for r in subset]),
        "mean_clip_T·I_prompted": mean([r["clip_text_image_prompted"] for r in subset]),
        "mean_clip_delta": mean([r["clip_delta_prompt_minus_uncond"] for r in subset]),
        "mean_lpips": mean([r["lpips_prompt_vs_uncond"] for r in subset]),
    }

print("\n=== Summary by guidance_scale ===")
for gs in GUIDANCE_SCALES:
    s = summary[gs]
    print(f"gs={gs:>4}:  T·I(uncond)={s['mean_clip_T·I_uncond']:.4f}  "
          f"T·I(prompt)={s['mean_clip_T·I_prompted']:.4f}  "
          f"Δ(T·I)={s['mean_clip_delta']:.4f}  LPIPS={s['mean_lpips']:.4f}")

# Write CSV
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {CSV_OUT} and saved images under {OUTDIR}/")
