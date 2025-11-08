
# Stable Diffusion 1.5 Fune-tuning for Spectrograms

This Python project fine-tunes Stable Diffusion 1.5 and Stable Diffusion XL 1.0 to produce spectrograms to produce 15 second novel audio clips using the Song Describer Dataset.

## Input Dataset

[Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset)

## Training and Validation

- Prepare Song Describer Dataset
  - Prepare mel spectrograms with 15 second offset and normalized loudness with text descriptions from dataset
- Update core training script to produce sample images (VAE decode) and print quality statistics (quick spectral entropy checks)
- LoRA fine-tune Stable Diffusion 1.5 and Stable Diffusion XL 1.0
- Evaluate model quality
  - Test prompt adherence and text encoder associations
  - Test diversity and check for model collapse
  - Audio reconstruction tests with img2img

Use ```AdamW``` optimizer. Some optimizers do not support text encoder training.

|Action|Result|
|----------------|------------------|
|Cache latents|✅ Faster training|
|Increase min SNR gamma|Indeterminate|
|Increase clip skip|Indeterminate|
|Increase rank and alpha per Thinking Machines Lab recommendation|✅ Improved definition|
|Shuffle captions|✅ Improved text encoder associations|
|Increase batch size|❌ Weaker prompt adherence|
|Increase dataset repeats|✅ Stronger prompt adherence|
|Increase steps for higher rank|✅ Improved text encoder associations and visual quality|
|Increase rank and alpha even further|❌ Mode collapse|
|Add small amount of noise|❌ Worse audio quality|
|Try Stable Diffusion XL 1.0 instead of 1.5|More visual details but similar audio quality; indeterminate|
|Disable bucketing|Indeterminate|
|SpecAugment some training images|Indeterminate|

Average Δ over 15 runs: 0.0197

Cosine probe 0.39379143714904785 to 0.9031552672386169