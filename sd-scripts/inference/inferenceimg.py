from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch, itertools, os

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "../tools/output_merged_sd",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,
).to(device)

# DPM++ 2M Karras
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)
if hasattr(pipe.scheduler, "solver_order"):
    pipe.scheduler.solver_order = 2
if hasattr(pipe.scheduler, "lower_order_final"):
    pipe.scheduler.lower_order_final = True

# Optional: memory/perf hints (safe to skip if you hit issues)
try:
    pipe.enable_attention_slicing("max")
except Exception:
    pass
try:
    pipe.enable_vae_tiling()
except Exception:
    pass

init_image = Image.open("../control/piano.png").convert("RGB")

prompt = (
    "This is an instrumental track based on electronic samples that can be used before the start of a movie to give a doomsday or post apocalypse feel mel spectrogram"
)

negative_prompt = (
    "text, watermark, blank, distorted"
)

# Cache embeddings to keep guidance consistent across runs
prompt_embeds, neg_embeds = pipe.encode_prompt(
    prompt=prompt,
    negative_prompt=negative_prompt,
    device=device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

# Try a small grid to find a good region of control
strength_list        = [0.25, 0.30, 0.35, 0.45, 0.55, 0.65]       # higher => more prompt control, less init faithfulness
guidance_scale_list  = [8.0, 10.0, 12.0, 14.0]        # more text steering up to ~14 (often sweet spot)

i = 0
for s, gs in itertools.product(strength_list, guidance_scale_list):
    gen = torch.Generator(device=device)#.manual_seed(1003) #0 + i)  # deterministic per combo
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        image=init_image,
        strength=s,
        num_inference_steps=40,      # a bit more steps helps adherence
        guidance_scale=gs,
        generator=gen,
    ).images[0]
    out_path = f"./s_{s:.2f}_gs_{gs:.1f}_2.png"
    image.save(out_path)
    print("saved:", out_path)
    i += 1
