from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "../tools/output_merged_sd",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None
).to("cuda" if torch.cuda.is_available() else "cpu")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
)
pipe.scheduler.config.solver_order = 2
pipe.scheduler.config.lower_order_final = True

negative = "stripes, striped, monotone"

# Generate images with prompt or unconditionally
for i in range(10):
    prompt = "Sad instrumental piano song that feels like is telling a story and that one can be put on in the background to listen while alone mel spectrogram"  # or use ""
    image = pipe(prompt, negative_prompt=negative, num_inference_steps=32, guidance_scale=4.0, height=256, width=472).images[0]
    image.save(f"spectrogram_{i+1}.png")
    print(f"Saved spectrogram_{i+1}.png")
