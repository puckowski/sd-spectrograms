
# Stable Diffusion 1.5 Fune-tuning for Spectrograms

This project fine-tunes Stable Diffusion 1.5 to produce spectrograms to produce 15 second novel audio clips using the Song Describer Dataset.

## Input Dataset

[Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset)

## Train Results

|Spectral Entropy|
|----------------|
|4.98 horizontal|
|6.93 vertical|
|6.93 overall|

| Summary by guidance_scale |
|---------------------------|
|gs= 0.0:  T·I(uncond)=0.1658  T·I(prompt)=0.1902  Δ(T·I)=0.0245  LPIPS=0.6472|
|gs= 2.0:  T·I(uncond)=0.1658  T·I(prompt)=0.2060  Δ(T·I)=0.0402  LPIPS=0.6819|
|gs= 4.0:  T·I(uncond)=0.1658  T·I(prompt)=0.2231  Δ(T·I)=0.0574  LPIPS=0.6976|
|gs= 8.0:  T·I(uncond)=0.1658  T·I(prompt)=0.2314  Δ(T·I)=0.0657  LPIPS=0.7061|
|gs=12.0:  T·I(uncond)=0.1658  T·I(prompt)=0.2265  Δ(T·I)=0.0607  LPIPS=0.7089|

| Prompt Steering |
|-----------------|
|Iter 1 | Base: 0.2111  Fine: 0.2219  Δ: 0.0108|
|Iter 2 | Base: 0.2362  Fine: 0.2311  Δ: -0.0051|
|Iter 3 | Base: 0.2106  Fine: 0.2272  Δ: 0.0167|
|Iter 4 | Base: 0.2148  Fine: 0.2180  Δ: 0.0032|
|Iter 5 | Base: 0.2143  Fine: 0.2275  Δ: 0.0132|
|Average Δ over 5 runs: 0.0098|

| Text Encoder Strength |
|-----------------------|
|cosine(base,ft) = 0.603907585144043   L2 shift = 14.993247985839844|
|cosine(base,ft) = 0.7494967579841614   L2 shift = 13.205038070678711|
