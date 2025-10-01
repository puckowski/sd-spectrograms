
# Stable Diffusion 1.5 Fune-tuning for Spectrograms

This Python project fine-tunes Stable Diffusion 1.5 to produce spectrograms to produce 15 second novel audio clips using the Song Describer Dataset.

## Input Dataset

[Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset)

## Training and Validation

- Prepare Song Describer Dataset
  - Prepare mel spectrograms with 15 second offset and normalized loudness with text descriptions from dataset
- Update core training script to produce sample images (VAE decode) and print quality statistics (quick spectral entropy checks)
- LoRA fine-tune Stable Diffusion 1.5
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

