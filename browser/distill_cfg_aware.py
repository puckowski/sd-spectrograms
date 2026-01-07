# distill_sdxl_student_kohya_mirror_with_real_latents_cfg.py
#
# Teacher -> Student UNet distillation for SDXL, with:
# - "kohya-mirror" training dynamics where applicable
# - timestep curriculum (bias toward low t early)
# - REAL-LATENT MIXING
# - CFG-AWARE DISTILLATION:
#     * teach student BOTH conditional and unconditional predictions
#     * optionally teach student to match teacher's guided prediction:
#         guided = uncond + cfg * (cond - uncond)
#
# Notes:
# - For SDXL, unconditional is typically "empty prompt".
# - This makes your student respond much better to CFG at inference.

import os
import glob
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, DDPMScheduler
from PIL import Image


# -----------------------------
# Config (paths / data)
# -----------------------------
model_path = "./tools/output_merged_sd"
captions_root = "images/10_song"
images_root = "images/10_song"
image_glob = "**/*.png"

save_dir = "distill_out_student_balanced_kohya_mirror_real_cfg"
save_every = 2000

H, W = 256, 512
LAT_H, LAT_W = H // 8, W // 8

max_steps = 400_000              # micro-steps (not optimizer steps)
batch_size = 1
seed = 1234

# -----------------------------
# Mirror kohya training dynamics (as applicable)
# -----------------------------
MIXED_PRECISION = "fp32"         # "fp16", "bf16", or "fp32"
NO_HALF_VAE = True               # keep VAE in fp32

GRAD_ACCUM_STEPS = 2

BASE_LR = 1e-4
WARMUP_FRAC = 0.02               # fraction of OPTIMIZER steps, not micro-steps

betas = (0.9, 0.99)
weight_decay = 1e-2
grad_clip = 1.0

t_lo = 50
t_hi_override = 800

MIN_SNR_GAMMA = 5.0
USE_MIN_SNR_WEIGHTING = True

NOISE_OFFSET = 0.05
NOISE_OFFSET_TYPE = "Original"

# -----------------------------
# Student capacity profile
# -----------------------------
STUDENT_PROFILE = "medium"       # "tiny" | "balanced" | "small" | "medium"

# -----------------------------
# Timestep curriculum (bias toward lower t)
# -----------------------------
CURRICULUM_STEPS = 80_000
T_BIAS_POWER = 3.0

# -----------------------------
# Distillation loss config
# -----------------------------
SMOOTH_L1_BETA = 0.01

# -----------------------------
# Real-latent mixing config
# -----------------------------
P_REAL_LATENTS = 0.70  # try 0.5-0.8 for spectrograms

CACHE_REAL_LATENTS_TO_DISK = True
REAL_LATENT_CACHE_DIR = "real_latent_cache_npy"
REAL_LATENT_RAM_CACHE_MAX = 256

# -----------------------------
# CFG distillation config (NEW)
# -----------------------------
# Teach student both uncond and cond, and (optionally) guided prediction too.
CFG_TRAIN_ENABLE = True

# Probability of using CFG-guided target (vs plain cond/uncond targets).
P_GUIDED_TARGET = 0.50

# Choose a CFG distribution that matches your intended inference.
# Strong steering typical range for distilled students: ~2-5.
CFG_MIN = 1.5
CFG_MAX = 5.0

# Also include a few low/near-1 samples so it doesn't break at low CFG.
CFG_P_LOW = 0.15     # chance to sample CFG in [1.0, 1.5]
CFG_LOW_MIN = 1.0
CFG_LOW_MAX = 1.5

# Relative weights for multi-target loss components.
LAMBDA_UNCOND = 1.0
LAMBDA_COND = 1.0
LAMBDA_GUIDED = 1.0


# -----------------------------
# Safe attention settings (PyTorch 2.x)
# -----------------------------
def force_safe_attention():
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["DIFFUSERS_USE_XFORMERS"] = "0"

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def stats(name: str, t: torch.Tensor):
    td = t.detach()
    print(
        f"[{name}] finite={torch.isfinite(td).all().item()} "
        f"shape={tuple(td.shape)} dtype={td.dtype} "
        f"min={td.min().item():.4g} max={td.max().item():.4g} "
        f"mean={td.mean().item():.4g} std={td.std().item():.4g}"
    )


def sample_t(step: int, device: str, t_lo_: int, t_hi_: int, batch: int) -> torch.Tensor:
    if step < CURRICULUM_STEPS:
        u = torch.rand((batch,), device=device)
        u = u ** T_BIAS_POWER
        t = (t_lo_ + (t_hi_ - t_lo_) * u).long()
    else:
        t = torch.randint(t_lo_, t_hi_ + 1, (batch,), device=device)
    return t


def apply_noise_offset(noise: torch.Tensor, offset: float):
    if offset is None or offset <= 0:
        return noise
    b, c, _, _ = noise.shape
    return noise + offset * torch.randn((b, c, 1, 1), device=noise.device, dtype=noise.dtype)


def move_scheduler_to_device(sched: DDPMScheduler, device: str):
    # Keep scheduler buffers in fp32 for stability
    if hasattr(sched, "alphas_cumprod") and isinstance(sched.alphas_cumprod, torch.Tensor):
        sched.alphas_cumprod = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    for name in ["betas", "alphas"]:
        if hasattr(sched, name) and isinstance(getattr(sched, name), torch.Tensor):
            setattr(sched, name, getattr(sched, name).to(device=device, dtype=torch.float32))


@torch.no_grad()
def teacher_distance(pred: torch.Tensor, teacher: torch.Tensor):
    a = pred.detach().float()
    b = teacher.detach().float()
    diff = a - b

    mse = (diff * diff).mean()
    rmse = torch.sqrt(mse + 1e-12)
    mae = diff.abs().mean()

    a_flat = a.view(a.shape[0], -1)
    b_flat = b.view(b.shape[0], -1)
    cos = F.cosine_similarity(a_flat, b_flat, dim=1).mean()

    return {"mse": mse.item(), "rmse": rmse.item(), "mae": mae.item(), "cos": cos.item()}


# -----------------------------
# Real latent caching helpers
# -----------------------------
class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        if self.max_size <= 0:
            return
        if key in self.data:
            self.data[key] = value
            return
        if len(self.data) >= self.max_size:
            self.data.pop(next(iter(self.data)))
        self.data[key] = value


def cache_path_for_image(img_path: str) -> str:
    safe = img_path.replace(":", "").replace("\\", "_").replace("/", "_")
    return os.path.join(REAL_LATENT_CACHE_DIR, safe + f"__{W}x{H}.npy")


def load_image_as_vae_input(path: str, H_: int, W_: int) -> torch.Tensor:
    """
    Returns float32 torch tensor [1,3,H,W] in [-1, 1], CPU tensor.
    """
    img = Image.open(path).convert("RGB")
    if img.size != (W_, H_):
        img = img.resize((W_, H_), resample=Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = x * 2.0 - 1.0
    return x


@torch.no_grad()
def image_to_latent(pipe: StableDiffusionXLPipeline, image_path: str, device: str, vae_dtype: torch.dtype) -> torch.Tensor:
    x = load_image_as_vae_input(image_path, H, W).to(device=device, dtype=vae_dtype)
    latents = pipe.vae.encode(x).latent_dist.sample()
    scale = getattr(pipe.vae.config, "scaling_factor", 0.13025)
    return latents * scale


def get_real_latent(
    pipe: StableDiffusionXLPipeline,
    img_path: str,
    device: str,
    vae_dtype: torch.dtype,
    unet_dtype: torch.dtype,
    ram_cache: Optional[LRUCache],
) -> torch.Tensor:
    key = img_path

    if ram_cache is not None:
        v = ram_cache.get(key)
        if v is not None:
            return v.to(device=device, dtype=unet_dtype)

    disk_path = cache_path_for_image(img_path)
    if CACHE_REAL_LATENTS_TO_DISK:
        os.makedirs(REAL_LATENT_CACHE_DIR, exist_ok=True)
        if os.path.exists(disk_path):
            arr = np.load(disk_path)  # float32 [4, LAT_H, LAT_W]
            t = torch.from_numpy(arr).unsqueeze(0)  # [1,4,H,W] CPU
            if ram_cache is not None:
                ram_cache.put(key, t)
            return t.to(device=device, dtype=unet_dtype)

    lat = image_to_latent(pipe, img_path, device=device, vae_dtype=vae_dtype)
    lat_cpu_fp32 = lat.detach().float().cpu()

    if CACHE_REAL_LATENTS_TO_DISK:
        np.save(disk_path, lat_cpu_fp32.squeeze(0).numpy())
    if ram_cache is not None:
        ram_cache.put(key, lat_cpu_fp32)

    return lat.to(device=device, dtype=unet_dtype)


# -----------------------------
# CFG helpers (NEW)
# -----------------------------
def sample_cfg() -> float:
    # Mixture: mostly strong range, sometimes low range (including ~1.0)
    if random.random() < CFG_P_LOW:
        return random.uniform(CFG_LOW_MIN, CFG_LOW_MAX)
    return random.uniform(CFG_MIN, CFG_MAX)


@torch.no_grad()
def encode_pair(pipe: StableDiffusionXLPipeline, prompt: str, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                                    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      (cond_prompt_embeds, cond_pooled, cond_time_ids,
       uncond_prompt_embeds, uncond_pooled, uncond_time_ids)
    """
    # conditional
    out_c = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    if len(out_c) == 2:
        c_prompt_embeds, c_pooled = out_c
    else:
        c_prompt_embeds = out_c[0]
        c_pooled = out_c[2]

    # unconditional (empty prompt)
    out_u = pipe.encode_prompt(
        prompt="",
        prompt_2="",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    if len(out_u) == 2:
        u_prompt_embeds, u_pooled = out_u
    else:
        u_prompt_embeds = out_u[0]
        u_pooled = out_u[2]

    # time ids (same for both)
    time_ids = torch.tensor([[H, W, 0, 0, H, W]], device=device, dtype=c_prompt_embeds.dtype)
    if time_ids.shape[0] != c_prompt_embeds.shape[0]:
        time_ids = time_ids.repeat(c_prompt_embeds.shape[0], 1)

    # Ensure uncond time_ids matches dtype
    u_time_ids = time_ids.to(dtype=u_prompt_embeds.dtype)

    return c_prompt_embeds, c_pooled, time_ids, u_prompt_embeds, u_pooled, u_time_ids


def main():
    force_safe_attention()
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and MIXED_PRECISION == "fp16":
        unet_dtype = torch.float16
    elif device == "cuda" and MIXED_PRECISION == "bf16":
        unet_dtype = torch.bfloat16
    elif MIXED_PRECISION == "fp32":
        unet_dtype = torch.float32
    else:
        unet_dtype = torch.float32

    vae_dtype = torch.float32 if NO_HALF_VAE else unet_dtype
    print(f"[info] device={device} unet_dtype={unet_dtype} vae_dtype={vae_dtype}")

    # -----------------------------
    # Load teacher
    # -----------------------------
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=unet_dtype,
        safety_checker=None,
    ).to(device)

    pipe.vae.to(device=device, dtype=vae_dtype)

    pipe.unet.eval()
    pipe.vae.eval()
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.eval()
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.eval()

    try:
        from diffusers.models.attention_processor import AttnProcessor
        pipe.unet.set_attn_processor(AttnProcessor())
        print("[info] teacher: set AttnProcessor()")
    except Exception as e:
        print("[info] teacher: could not set AttnProcessor (ok):", repr(e))

    sched = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    move_scheduler_to_device(sched, device=device)

    num_train_timesteps = sched.config.num_train_timesteps
    t_hi = min(t_hi_override, num_train_timesteps - 1)

    # local (NO global mutation)
    p_real_latents = float(P_REAL_LATENTS)

    print(f"[info] scheduler timesteps: 0..{num_train_timesteps-1} using train range {t_lo}..{t_hi}")
    print(f"[info] latent size: (B,4,{LAT_H},{LAT_W}) for {W}x{H}")
    print(f"[info] curriculum: steps<{CURRICULUM_STEPS} bias_power={T_BIAS_POWER}, then uniform")
    print(f"[info] min_snr_gamma={MIN_SNR_GAMMA} noise_offset={NOISE_OFFSET} ({NOISE_OFFSET_TYPE})")
    if CFG_TRAIN_ENABLE:
        print(f"[info] CFG training enabled: cfg~U([{CFG_MIN},{CFG_MAX}]) "
              f"+ low-mix p={CFG_P_LOW} U([{CFG_LOW_MIN},{CFG_LOW_MAX}]) "
              f"p_guided={P_GUIDED_TARGET}")

    # -----------------------------
    # Captions
    # -----------------------------
    txt_files = sorted(glob.glob(os.path.join(captions_root, "**", "*.txt"), recursive=True))
    if not txt_files:
        raise FileNotFoundError(f"No .txt captions found under: {captions_root}")

    captions: List[str] = []
    for p in txt_files:
        text = Path(p).read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            captions.append(text)

    if not captions:
        raise RuntimeError(f"Found {len(txt_files)} .txt files but all were empty.")

    print(f"[info] loaded {len(captions)} captions from {captions_root}")

    # -----------------------------
    # Real images list
    # -----------------------------
    img_files = sorted(glob.glob(os.path.join(images_root, image_glob), recursive=True))
    if img_files:
        print(f"[info] found {len(img_files)} real images under {images_root}/{image_glob}")
    else:
        print(f"[warn] no images found under {images_root}/{image_glob} -> disabling real-latent mixing")
        p_real_latents = 0.0

    print(f"[info] real-latent mixing: p_real={p_real_latents} cache_disk={CACHE_REAL_LATENTS_TO_DISK}")
    print("[info] distillation: teacher-match (CFG-aware)")

    real_latent_cache = LRUCache(REAL_LATENT_RAM_CACHE_MAX) if REAL_LATENT_RAM_CACHE_MAX > 0 else None

    # -----------------------------
    # Build student config
    # -----------------------------
    teacher_cfg = dict(pipe.unet.config)

    down_types = list(teacher_cfg.get("down_block_types", []))
    up_types = list(teacher_cfg.get("up_block_types", []))
    if not down_types or not up_types:
        raise RuntimeError("Teacher config missing down_block_types/up_block_types")

    n_stages = min(len(down_types), len(up_types))
    if n_stages < 2:
        raise RuntimeError(f"Unexpected stage count from teacher: down={len(down_types)} up={len(up_types)}")

    down_types = down_types[:n_stages]
    up_types = up_types[:n_stages]

    teacher_cfg["sample_size"] = [LAT_H, LAT_W]
    teacher_cfg["addition_embed_type"] = teacher_cfg.get("addition_embed_type", "text_time")
    teacher_cfg["addition_time_embed_dim"] = teacher_cfg.get("addition_time_embed_dim", 256)
    teacher_cfg["cross_attention_dim"] = teacher_cfg.get("cross_attention_dim", 2048)

    if STUDENT_PROFILE == "tiny":
        base = [128, 256, 384, 512]
        teacher_cfg["layers_per_block"] = 1
        teacher_cfg["transformer_layers_per_block"] = 1
        teacher_cfg["attention_head_dim"] = 64
    elif STUDENT_PROFILE == "balanced":
        base = [160, 288, 416, 544]  # divisible by 32
        teacher_cfg["layers_per_block"] = 2
        teacher_cfg["transformer_layers_per_block"] = 1
        teacher_cfg["attention_head_dim"] = 64
    elif STUDENT_PROFILE == "small":
        base = [160, 320, 480, 640]
        teacher_cfg["layers_per_block"] = 2
        teacher_cfg["transformer_layers_per_block"] = 1
        teacher_cfg["attention_head_dim"] = 64
    elif STUDENT_PROFILE == "medium":
        # Slightly larger than "small"
        base = [192, 384, 576, 768]  # divisible by 32
        teacher_cfg["layers_per_block"] = 2
        teacher_cfg["transformer_layers_per_block"] = 1
        teacher_cfg["attention_head_dim"] = 64
    else:
        raise ValueError("STUDENT_PROFILE must be 'tiny', 'balanced', 'small', or 'medium'")

    teacher_cfg["block_out_channels"] = base[:n_stages]
    teacher_cfg["down_block_types"] = down_types
    teacher_cfg["up_block_types"] = up_types

    print("[info] teacher stages:", n_stages)
    print("[info] student block_out_channels:", teacher_cfg["block_out_channels"])

    student = UNet2DConditionModel(**teacher_cfg).to(device=device, dtype=unet_dtype).train()

    try:
        from diffusers.models.attention_processor import AttnProcessor
        student.set_attn_processor(AttnProcessor())
        print("[info] student: set AttnProcessor()")
    except Exception:
        pass

    def init_small(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    student.apply(init_small)

    opt = torch.optim.AdamW(student.parameters(), lr=BASE_LR, betas=betas, weight_decay=weight_decay)

    total_opt_steps = max(1, max_steps // GRAD_ACCUM_STEPS)
    warmup_steps = int(WARMUP_FRAC * total_opt_steps)
    cosine_steps = max(1, total_opt_steps - warmup_steps)

    print(
        f"[info] grad_accum={GRAD_ACCUM_STEPS} total_opt_steps={total_opt_steps} "
        f"warmup_steps={warmup_steps} cosine_steps={cosine_steps}"
    )

    def set_lr(new_lr: float):
        for pg in opt.param_groups:
            pg["lr"] = float(new_lr)

    os.makedirs(save_dir, exist_ok=True)
    opt.zero_grad(set_to_none=True)

    for step in range(max_steps):
        prompt = random.choice(captions)

        # NEW: encode both cond & uncond
        if CFG_TRAIN_ENABLE:
            c_emb, c_pool, c_time, u_emb, u_pool, u_time = encode_pair(pipe, prompt, device)
        else:
            # fallback: only conditional
            out = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            if len(out) == 2:
                c_emb, c_pool = out
            else:
                c_emb, c_pool = out[0], out[2]
            c_time = torch.tensor([[H, W, 0, 0, H, W]], device=device, dtype=c_emb.dtype)

            u_emb, u_pool, u_time = None, None, None

        use_real = (p_real_latents > 0.0) and (random.random() < p_real_latents) and bool(img_files)
        if use_real:
            img_path = random.choice(img_files)
            latents = get_real_latent(
                pipe, img_path,
                device=device,
                vae_dtype=vae_dtype,
                unet_dtype=unet_dtype,
                ram_cache=real_latent_cache,
            )
            if batch_size != 1:
                latents = latents.repeat(batch_size, 1, 1, 1)
        else:
            latents = torch.randn((batch_size, 4, LAT_H, LAT_W), device=device, dtype=unet_dtype)

        t = sample_t(step, device, t_lo, t_hi, batch_size)

        noise = torch.randn_like(latents)
        noise = apply_noise_offset(noise, NOISE_OFFSET)
        x_t = sched.add_noise(latents, noise, t)

        # -----------------------------
        # Teacher targets (CFG-aware)
        # -----------------------------
        if CFG_TRAIN_ENABLE:
            with torch.no_grad():
                teacher_cond = pipe.unet(
                    x_t, t,
                    encoder_hidden_states=c_emb,
                    added_cond_kwargs={"text_embeds": c_pool, "time_ids": c_time},
                ).sample

                teacher_uncond = pipe.unet(
                    x_t, t,
                    encoder_hidden_states=u_emb,
                    added_cond_kwargs={"text_embeds": u_pool, "time_ids": u_time},
                ).sample

                cfg = sample_cfg()
                teacher_guided = teacher_uncond + float(cfg) * (teacher_cond - teacher_uncond)

        else:
            with torch.no_grad():
                teacher_cond = pipe.unet(
                    x_t, t,
                    encoder_hidden_states=c_emb,
                    added_cond_kwargs={"text_embeds": c_pool, "time_ids": c_time},
                ).sample
            teacher_uncond = None
            teacher_guided = None
            cfg = None

        # -----------------------------
        # Student predictions (CFG-aware)
        # -----------------------------
        if CFG_TRAIN_ENABLE:
            pred_cond = student(
                x_t, t,
                encoder_hidden_states=c_emb,
                added_cond_kwargs={"text_embeds": c_pool, "time_ids": c_time},
            ).sample

            pred_uncond = student(
                x_t, t,
                encoder_hidden_states=u_emb,
                added_cond_kwargs={"text_embeds": u_pool, "time_ids": u_time},
            ).sample

            pred_guided = pred_uncond + float(cfg) * (pred_cond - pred_uncond)
        else:
            pred_cond = student(
                x_t, t,
                encoder_hidden_states=c_emb,
                added_cond_kwargs={"text_embeds": c_pool, "time_ids": c_time},
            ).sample
            pred_uncond = None
            pred_guided = None

        # -----------------------------
        # Sanity checks
        # -----------------------------
        if not torch.isfinite(teacher_cond).all():
            stats("x_t", x_t); stats("teacher_cond", teacher_cond)
            raise RuntimeError("Teacher (cond) produced NaNs/Infs")
        if not torch.isfinite(pred_cond).all():
            stats("pred_cond", pred_cond); stats("teacher_cond", teacher_cond)
            raise RuntimeError("Student (cond) produced NaNs/Infs")

        if CFG_TRAIN_ENABLE:
            if not torch.isfinite(teacher_uncond).all():
                stats("teacher_uncond", teacher_uncond)
                raise RuntimeError("Teacher (uncond) produced NaNs/Infs")
            if not torch.isfinite(pred_uncond).all():
                stats("pred_uncond", pred_uncond)
                raise RuntimeError("Student (uncond) produced NaNs/Infs")
            if not torch.isfinite(teacher_guided).all():
                stats("teacher_guided", teacher_guided)
                raise RuntimeError("Teacher (guided) produced NaNs/Infs")
            if not torch.isfinite(pred_guided).all():
                stats("pred_guided", pred_guided)
                raise RuntimeError("Student (guided) produced NaNs/Infs")

        # -----------------------------
        # Loss (CFG-aware)
        # -----------------------------
        def smoothl1_per_elem(a, b):
            return F.smooth_l1_loss(a, b, beta=SMOOTH_L1_BETA, reduction="none")

        # base losses
        per_elem_cond = smoothl1_per_elem(pred_cond, teacher_cond)

        if CFG_TRAIN_ENABLE:
            per_elem_uncond = smoothl1_per_elem(pred_uncond, teacher_uncond)

            # sometimes include guided target (to make student match teacher guidance behavior directly)
            use_guided = (random.random() < P_GUIDED_TARGET)
            if use_guided:
                per_elem_guided = smoothl1_per_elem(pred_guided, teacher_guided)
            else:
                per_elem_guided = None
        else:
            per_elem_uncond = None
            per_elem_guided = None
            use_guided = False

        # min-SNR weighting
        if USE_MIN_SNR_WEIGHTING:
            alpha_bar = sched.alphas_cumprod[t].view(-1, 1, 1, 1)  # fp32
            snr = alpha_bar / (1.0 - alpha_bar)
            gamma = torch.tensor(MIN_SNR_GAMMA, device=device, dtype=snr.dtype)
            w = torch.minimum(snr, gamma) / (snr + 1e-8)
            w = w.to(per_elem_cond.dtype)

            per_elem_cond = per_elem_cond * w
            if CFG_TRAIN_ENABLE:
                per_elem_uncond = per_elem_uncond * w
                if per_elem_guided is not None:
                    per_elem_guided = per_elem_guided * w

        # combine
        loss = per_elem_cond.mean()

        if CFG_TRAIN_ENABLE:
            loss = loss + (LAMBDA_UNCOND * per_elem_uncond.mean())
            if per_elem_guided is not None:
                loss = loss + (LAMBDA_GUIDED * per_elem_guided.mean())

            # optionally scale down total when guided is absent (keeps loss magnitude steadier)
            denom = (LAMBDA_COND + LAMBDA_UNCOND + (LAMBDA_GUIDED if per_elem_guided is not None else 0.0))
            loss = loss / max(1e-8, denom)

        if not torch.isfinite(loss):
            print("[err] loss became non-finite")
            stats("pred_cond", pred_cond); stats("teacher_cond", teacher_cond)
            if CFG_TRAIN_ENABLE:
                stats("pred_uncond", pred_uncond); stats("teacher_uncond", teacher_uncond)
            raise RuntimeError("Loss became NaN/Inf")

        (loss / GRAD_ACCUM_STEPS).backward()

        do_opt_step = ((step + 1) % GRAD_ACCUM_STEPS == 0)
        if do_opt_step:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)

            opt_step = (step + 1) // GRAD_ACCUM_STEPS
            if warmup_steps > 0 and opt_step <= warmup_steps:
                set_lr(BASE_LR * (opt_step / warmup_steps))
            else:
                k = opt_step - warmup_steps
                k = min(max(k, 0), cosine_steps)
                cos = 0.5 * (1.0 + torch.cos(torch.tensor(float(k) * 3.141592653589793 / float(cosine_steps))))
                set_lr(BASE_LR * float(cos))

            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            mode = "curric" if step < CURRICULUM_STEPS else "uniform"
            cur_lr = opt.param_groups[0]["lr"]
            src = "real" if use_real else "rand"

            # log dist on conditional path (always valid)
            dist_c = teacher_distance(pred_cond, teacher_cond)
            extra = ""
            if CFG_TRAIN_ENABLE:
                dist_u = teacher_distance(pred_uncond, teacher_uncond)
                extra = f" cfg={cfg:.2f} guided={'Y' if use_guided else 'N'} distU_rmse={dist_u['rmse']:.6e} distU_cos={dist_u['cos']:.6f}"

            print(
                f"step={step:07d} loss={loss.item():.6f} lr={cur_lr:.3e} src={src} "
                f"distC_rmse={dist_c['rmse']:.6e} distC_cos={dist_c['cos']:.6f} "
                f"t=[{int(t.min())},{int(t.max())}] mode={mode} "
                f"prompt='{prompt[:80]}'{extra}"
            )

        if save_every and step > 0 and step % save_every == 0:
            ckpt_dir = os.path.join(save_dir, f"student_step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            student.save_pretrained(ckpt_dir)
            print(f"[save] wrote {ckpt_dir}")

    final_dir = os.path.join(save_dir, "student_final")
    os.makedirs(final_dir, exist_ok=True)
    student.save_pretrained(final_dir)
    print(f"[done] saved final student to {final_dir}")


if __name__ == "__main__":
    main()
