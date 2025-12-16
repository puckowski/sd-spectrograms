# DiffusionMiniMIDI

## Stable Diffusion 1.5 Fine-tuning for Spectrograms

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
  - Test diversity and check for mode collapse
  - Audio reconstruction tests with img2img

Use ```AdamW``` optimizer. Some optimizers do not support text encoder training.

|Action|Result|
|----------------|------------------|
|Cache latents|✅ Faster training|
|Increase min SNR gamma|Indeterminate|
|Increase clip skip|Indeterminate|
|Increase rank and alpha per Thinking Machines Lab recommendation|✅ Improved definition|
|Shuffle captions|✅ Improved text encoder associations|
|Increase batch size|✅ Prevent mode collapse while adhering to prompts|
|Increase dataset repeats|✅ Stronger prompt adherence|
|Increase steps for higher rank|✅ Improved text encoder associations and visual quality|
|Increase rank and alpha even further|❌ Mode collapse|
|Add small amount of noise|❌ Worse audio quality|
|Try Stable Diffusion XL 1.0 instead of 1.5|More visual details but similar audio quality. Slower training and inference; indeterminate|
|Disable bucketing|Indeterminate|
|SpecAugment some training images|Indeterminate|
|Limit dataset to instrumental songs|✅ Improved audio quality for instrumental clips|
|Add regularization images|✅ Improved inference variety and quality|
|Train with Huber loss|✅ Improved inference detail|
|Train with spectrogram-aware loss|Experimenting|
|Resume training with different hyperparameters|Experimenting|

Pixel RMSE diversity:
  Mean RMSE across pairs: 0.207607

Spectral entropy diversity:
  Mean entropy: 11.649665
  Std entropy : 0.019429

CLIP Embedding diversity:
  Mean pairwise cosine distance: 0.037015
  Covariance trace: 0.037015

*** Concept: piano ***
mean sim forbidden (NO negative prompt)   = 0.2945
mean sim forbidden (WITH negative prompt) = 0.2743
delta (no_neg - with_neg)                 = 0.0202
std(no_neg), std(with_neg)                = 0.0107, 0.0150

--- text_encoder (ViT-L/14) most changed by cosine ---
[('<|endoftext|>', 0.6981689929962158, 22.547687530517578), ('would</w>', 0.6997650861740112, 21.954866409301758), ('instrumental</w>', 0.7992724180221558, 17.469383239746094), ('gram</w>', 0.8069127202033997, 17.298229217529297), ('a</w>', 0.827735185623169, 17.015817642211914), ('with</w>', 0.8432062268257141, 15.750553131103516), ('in</w>', 0.8514582514762878, 15.214371681213379), (',</w>', 0.8528823852539062, 15.233933448791504), ('mel</w>', 0.8548272848129272, 15.117595672607422), ('used</w>', 0.8557292222976685, 15.384247779846191), ('setting</w>', 0.8559130430221558, 14.844605445861816), ('the</w>', 0.8638472557067871, 14.606560707092285), ('spectro', 0.8682229518890381, 14.42866039276123), ('track</w>', 0.8720040321350098, 14.05140495300293), ('zers</w>', 0.8769415616989136, 13.605331420898438), ('of</w>', 0.8854455947875977, 13.574203491210938), ('synthesi', 0.897377610206604, 12.585013389587402), ('use</w>', 0.9019694328308105, 12.506190299987793), ('be</w>', 0.9025022983551025, 12.477262496948242), ('track</w>', 0.9140052199363708, 11.351520538330078), ('heavy</w>', 0.9206768870353699, 11.13010311126709), ('game</w>', 0.9418706893920898, 9.481133460998535), ('energetic</w>', 0.944836437702179, 9.279662132263184), ('video</w>', 0.9458879232406616, 9.111418724060059)]

--- text_encoder_2 (OpenCLIP bigG) most changed by cosine ---
[('<|endoftext|>', 0.16627950966358185, 64.28463745117188), ('with</w>', 0.3933849632740021, 49.198699951171875), ('the</w>', 0.3957698345184326, 49.414466857910156), ('a</w>', 0.40238356590270996, 46.7076530456543), ('be</w>', 0.40888041257858276, 47.71257400512695), ('used</w>', 0.4217931032180786, 47.70881271362305), ('in</w>', 0.42566394805908203, 48.782073974609375), ('gram</w>', 0.42830654978752136, 48.62892532348633), ('of</w>', 0.42996925115585327, 48.46480178833008), ('would</w>', 0.437828004360199, 49.26106643676758), ('mel</w>', 0.44340917468070984, 47.15966033935547), ('use</w>', 0.4479261636734009, 46.6311149597168), ('setting</w>', 0.4532841444015503, 47.594730377197266), ('spectro', 0.45960533618927, 49.41953659057617), (',</w>', 0.4802139103412628, 44.95549011230469), ('zers</w>', 0.48744678497314453, 46.99721908569336), ('instrumental</w>', 0.4934302568435669, 44.817745208740234), ('video</w>', 0.5214934945106506, 46.64348602294922), ('track</w>', 0.5344747304916382, 43.61231231689453), ('game</w>', 0.5393884778022766, 45.53972625732422), ('synthesi', 0.5447021722793579, 44.318870544433594), ('heavy</w>', 0.5553856492042542, 44.88984298706055), ('track</w>', 0.5654393434524536, 44.38675308227539), ('energetic</w>', 0.5852389335632324, 42.40277099609375)]

Pooled deltas:
text_encoder pooled: {'cosine': 0.6981689929962158, 'l2': 22.547685623168945}
text_encoder_2 pooled: {'cosine': 0.2159281224012375, 'l2': 51.77677917480469}
