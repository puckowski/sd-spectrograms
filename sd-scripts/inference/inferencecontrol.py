import os
from glob import glob
from typing import Tuple
import numpy as np
from PIL import Image
import torch

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

# ---------------- Config ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "./tools/output_merged_sd"
INPUT_DIR  = "control"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT  = "Instrumental electrodance track with a straight drum machine beat and a vocal sampler at the start and a repetitive synth chord progression with a pointy synth lead in the middle mel spectrogram"
NEGATIVE = "hiss, metallic, chirp, ringing, aliasing, oversharpen, banding, artifacts, color noise, washed out"

NUM_STEPS      = 60     # 40–60 helps ControlNet track source
GUIDANCE_SCALE = 1.8    # lower => more faithful to image; try 1.8–3.5
SEED           = None   # or None

TILE_SCALE   = 1.8      # 1.0–2.0 (increase for tighter match)
CANNY_SCALE  = 1.1      # 0.7–1.5
CTRL_START, CTRL_END = 0.0, 1.0

# -------------- Helpers --------------
def snap_to_eight(w: int, h: int) -> Tuple[int, int]:
    return w - (w % 8), h - (h % 8)

def canny_edges(pil_img: Image.Image, low=120, high=260) -> Image.Image:
    try:
        import cv2
        arr = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
    except Exception:
        # Fallback gradient magnitude
        arr = np.array(pil_img.convert("L"), dtype=np.float32)
        gx = np.zeros_like(arr); gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
        gy = np.zeros_like(arr); gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
        mag = np.sqrt(gx*gx + gy*gy)
        edges = (255.0 * (mag / (mag.max() + 1e-6))).astype(np.uint8)
    edges_rgb = np.stack([edges]*3, axis=-1)
    return Image.fromarray(edges_rgb)

# -------------- Load ControlNets --------------
controlnet_tile = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

# Try to import MultiControlNetModel if available
try:
    from diffusers import MultiControlNetModel  # newer versions
    multi_control_supported = True
    multi_control = MultiControlNetModel([controlnet_tile, controlnet_canny])
    controlnet_for_pipe = multi_control
    multi_mode = "multi_class"
except Exception:
    multi_control_supported = False
    multi_mode = "unknown"
    controlnet_for_pipe = controlnet_tile  # temporary; we may replace below

# -------------- Build Pipeline --------------
pipe = None
err = None
try:
    # First attempt: if we have MultiControlNetModel, use it
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_PATH,
        controlnet=controlnet_for_pipe,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    used_mode = multi_mode
except TypeError as e:
    err = e

# If that failed and we *don't* have MultiControlNetModel, try passing a list (supported by some versions)
if pipe is None and not multi_control_supported:
    try:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            MODEL_PATH,
            controlnet=[controlnet_tile, controlnet_canny],  # some versions accept a list directly
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        used_mode = "list_controls"
    except Exception as e:
        err = e

# Final fallback: single ControlNet (tile only)
if pipe is None:
    # Let the user know we fell back
    print("⚠️ Multi-ControlNet not supported in your diffusers build; falling back to Tile only.")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_PATH,
        controlnet=controlnet_tile,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    used_mode = "single_tile"

pipe = pipe.to(device)

# Memory/perf tweaks
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass
pipe.enable_attention_slicing("max")
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# Scheduler: DPM++ 2M Karras
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
)
pipe.scheduler.config.solver_order = 2
pipe.scheduler.config.lower_order_final = True

# Seed
generator = None
if SEED is not None:
    generator = torch.Generator(device=device).manual_seed(SEED)

# -------------- Inference --------------
for i in range(10):
    for img_path in glob(os.path.join(INPUT_DIR, "*.png")):
        try:
            src = Image.open(img_path).convert("RGB")
            w, h = snap_to_eight(*src.size)
            if (w, h) != src.size:
                src = src.resize((w, h), Image.BICUBIC)

            # Build control images/params depending on mode
            if used_mode in ("multi_class", "list_controls"):
                cn_images = [src, canny_edges(src)]
                cn_scales = [TILE_SCALE, CANNY_SCALE]
                cn_starts = [CTRL_START, CTRL_START]
                cn_ends   = [CTRL_END,   CTRL_END]
                result = pipe(
                    prompt=PROMPT,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    width=w, height=h,
                    control_image=cn_images,
                    controlnet_conditioning_scale=cn_scales,
                    control_guidance_start=cn_starts,
                    control_guidance_end=cn_ends,
                    generator=generator,
                )
            else:
                # single Tile fallback
                result = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    width=w, height=h,
                    image=src,  # some versions use 'image' or 'control_image' for single CN; support both
                    control_image=src,
                    controlnet_conditioning_scale=TILE_SCALE,
                    control_guidance_start=CTRL_START,
                    control_guidance_end=CTRL_END,
                    generator=generator,
                )

            image = result.images[0]
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(OUTPUT_DIR, f"{base}_gen_{i}.png")
            image.save(out_path)
            print(f"✅ [{used_mode}] {img_path} → {out_path}")
        except Exception as e:
            print(f"❌ Failed {img_path}: {e}")
