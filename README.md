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
|Dynamic huber loss starting at 0.15 ending at 0.05|✅ Improved inference detail|

=== CLAP text–audio similarity (single prompt, ALL outputs) ===
Mean CLAP similarity: 0.3740
Std  CLAP similarity: 0.0520

=== Embedding diversity (CLAP audio, ALL inputs/outputs) ===
Generated diversity: {'cov_trace': 0.11994156825213592, 'mean_pairwise_cos_dist': 0.11994153261184692}

=== FAD (Fréchet Audio Distance, CLAP space; ALL inputs vs ALL outputs) === 
FAD: 0.605529

LAION-CLAP text↔audio similarity scores
Old "a violin" 0.038578  
Old full prompt 0.220306
New "a violin" 0.124678
New full prompt 0.448420 

======================================
Evaluating negative prompts for concept: piano
======================================
seed=00 | sim_forbidden(no_neg)=0.2746 | sim_forbidden(with_neg)=0.2125
seed=01 | sim_forbidden(no_neg)=0.2870 | sim_forbidden(with_neg)=0.2677
seed=02 | sim_forbidden(no_neg)=0.2528 | sim_forbidden(with_neg)=0.2868
seed=03 | sim_forbidden(no_neg)=0.2828 | sim_forbidden(with_neg)=0.2512
seed=04 | sim_forbidden(no_neg)=0.2192 | sim_forbidden(with_neg)=0.2334
seed=05 | sim_forbidden(no_neg)=0.2869 | sim_forbidden(with_neg)=0.2467
seed=06 | sim_forbidden(no_neg)=0.2561 | sim_forbidden(with_neg)=0.2762
seed=07 | sim_forbidden(no_neg)=0.2672 | sim_forbidden(with_neg)=0.2814
seed=08 | sim_forbidden(no_neg)=0.2775 | sim_forbidden(with_neg)=0.2444
seed=09 | sim_forbidden(no_neg)=0.2693 | sim_forbidden(with_neg)=0.2487

--- text_encoder (ViT-L/14) most changed by cosine ---
[('<|endoftext|>', 0.46267402172088623, 30.106826782226562), ('synthesi', 0.6426379680633545, 23.31144905090332), ('instrumental</w>', 0.6697717308998108, 22.484073638916016), ('would</w>', 0.6778603792190552, 22.801740646362305), ('zers</w>', 0.698783278465271, 21.387943267822266), (',</w>', 0.7014576196670532, 21.755701065063477), ('used</w>', 0.7124971747398376, 21.866945266723633), ('with</w>', 0.7149113416671753, 21.230161666870117), ('in</w>', 0.717484712600708, 21.059389114379883), ('track</w>', 0.7251315712928772, 20.622774124145508), ('of</w>', 0.7316849231719971, 20.63111686706543), ('use</w>', 0.7387391328811646, 20.320722579956055), ('a</w>', 0.7549930810928345, 20.201135635375977), ('setting</w>', 0.7551804184913635, 19.36915397644043), ('be</w>', 0.7831117510795593, 18.644804000854492), ('the</w>', 0.791735053062439, 17.974346160888672), ('gram</w>', 0.7929743528366089, 17.86935806274414), ('mel</w>', 0.7962362766265869, 17.96268653869629), ('game</w>', 0.8104720115661621, 17.049026489257812), ('spectro', 0.8125004172325134, 17.148801803588867), ('track</w>', 0.8228615522384644, 16.328243255615234), ('energetic</w>', 0.8617182970046997, 14.64250373840332), ('video</w>', 0.8666972517967224, 14.268715858459473), ('heavy</w>', 0.8697217702865601, 14.293301582336426)]

--- text_encoder_2 (OpenCLIP bigG) most changed by cosine ---
[('<|endoftext|>', 0.09304465353488922, 72.23995208740234), ('a</w>', 0.2517144978046417, 55.24578094482422), ('with</w>', 0.26886415481567383, 55.84648132324219), ('use</w>', 0.3311299681663513, 54.879329681396484), ('spectro', 0.3346003293991089, 57.55559539794922), ('be</w>', 0.33546215295791626, 53.071834564208984), ('used</w>', 0.34105491638183594, 53.723968505859375), ('in</w>', 0.34203577041625977, 55.05217742919922), ('of</w>', 0.35115423798561096, 54.792240142822266), ('would</w>', 0.35408473014831543, 55.54985809326172), ('gram</w>', 0.3631989359855652, 53.979095458984375), (',</w>', 0.36532631516456604, 50.93788146972656), ('the</w>', 0.3683058023452759, 52.84638977050781), ('zers</w>', 0.3714083135128021, 54.61543655395508), ('mel</w>', 0.38767731189727783, 52.09356689453125), ('setting</w>', 0.39311718940734863, 52.192344665527344), ('video</w>', 0.40871602296829224, 54.69394302368164), ('heavy</w>', 0.4316938817501068, 53.07711410522461), ('game</w>', 0.44173792004585266, 52.606231689453125), ('synthesi', 0.4545367658138275, 50.453529357910156), ('an</w>', 0.475343257188797, 48.4663200378418), ('track</w>', 0.48489314317703247, 50.70996856689453), ('track</w>', 0.49969419836997986, 47.312435150146484), ('energetic</w>', 0.5106618404388428, 47.945091247558594)]

Pooled deltas:
text_encoder pooled: {'cosine': 0.46267402172088623, 'l2': 30.106826782226562}
text_encoder_2 pooled: {'cosine': 0.10787181556224823, 'l2': 57.664154052734375}
