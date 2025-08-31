# pip install torch torchvision pillow open_clip_torch diffusers==0.25.0
import torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_pipe(model_path):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )
    return pipe

# A: baseline (frozen TE)
pipe_base = make_pipe("runwayml/stable-diffusion-v1-5")

# B: your fine-tuned model (merged or base+LoRA)
pipe_ft   = make_pipe("../tools/output_merged_sd")

prompt = "an energetic instrumental track with heavy use of synthesizers, the track would be used in a video game setting mel spectrogram"

# -------- CLIP score setup ----------
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
model = model.to(device).eval()
tok = open_clip.get_tokenizer("ViT-L-14")

def clip_score(pipe, prompt, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    img = pipe(prompt, num_inference_steps=50, guidance_scale=8.0, generator=g).images[0]
    with torch.no_grad():
        im = preprocess(img).unsqueeze(0).to(device)
        txt = tok([prompt]).to(device)
        im_feats = model.encode_image(im)
        txt_feats = model.encode_text(txt)
        im_feats = im_feats / im_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        sim = (im_feats * txt_feats).sum(dim=-1)
    return sim.item()

# -------- Loop and average ----------
deltas = []
for i in range(5):
    seed = 1000 + i
    score_base = clip_score(pipe_base, prompt, seed)
    score_ft   = clip_score(pipe_ft,   prompt, seed)
    delta = score_ft - score_base
    deltas.append(abs(delta))
    print(f"Iter {i+1} | Base: {score_base:.4f}  Fine: {score_ft:.4f}  Δ: {delta:.4f}")

avg_delta = sum(deltas) / len(deltas)
print(f"\nAverage Δ over {len(deltas)} runs: {avg_delta:.4f}")
