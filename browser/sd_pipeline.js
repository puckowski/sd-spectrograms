import ort from "./ort.webgpu.min.mjs";
import { tokenize } from "./tokenizer.js";
import { DDIMScheduler, loadSchedulerConfig } from "./ddim_scheduler.js";

const H = 256, W = 512;
const LAT_H = H / 8, LAT_W = W / 8;  // 32, 64
const MAX_LEN = 77;

// Float16 helper
const hasFloat16 = typeof Float16Array !== "undefined";

// Clamp array values to prevent NaN/Inf
function clampArray(arr, min = -65504, max = 65504) {
  for (let i = 0; i < arr.length; i++) {
    const val = arr[i];
    if (isNaN(val) || !isFinite(val)) {
      arr[i] = 0;
    } else if (val < min) {
      arr[i] = min;
    } else if (val > max) {
      arr[i] = max;
    }
  }
  return arr;
}

function tensorF32(f32, dims) {
  return new ort.Tensor("float32", f32, dims);
}

function tensorF16OrF32(f32, dims) {
  if (hasFloat16) {
    const f16 = new Float16Array(f32.length);
    for (let i = 0; i < f32.length; i++) f16[i] = f32[i];
    return new ort.Tensor("float16", f16, dims);
  }
  return new ort.Tensor("float32", f32, dims);
}

function tensorI32(i32, dims) {
  return new ort.Tensor("int32", i32, dims);
}

function tensorI64FromInt32Array(i32, dims) {
  const out = new BigInt64Array(i32.length);
  for (let i = 0; i < i32.length; i++) out[i] = BigInt(i32[i]);
  return new ort.Tensor("int64", out, dims);
}

function tensorI64FromInt(t, dims = [1]) {
  // ORT Web uses BigInt64Array for int64
  return new ort.Tensor("int64", BigInt64Array.from([BigInt(t)]), dims);
}

// Simple seeded RNG + normal noise
function randn(n, seed = 0) {
  let s = (seed >>> 0) || 123456789;
  const rnd = () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };

  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rnd(), 1e-7);
    const u2 = rnd();
    const r = Math.sqrt(-2 * Math.log(u1));
    const th = 2 * Math.PI * u2;
    out[i] = r * Math.cos(th);
    if (i + 1 < n) out[i + 1] = r * Math.sin(th);
  }
  return out;
}

// concat TE1 + TE2 sequence embeddings -> [1,77,2048]
function concatLastDim(te1Seq, te2Seq) {
  const [B, T, D1] = te1Seq.dims;
  const [B2, T2, D2] = te2Seq.dims;
  if (B !== B2 || T !== T2) throw new Error("text encoder seq dims mismatch");

  const a = te1Seq.data;
  const b = te2Seq.data;

  const out = new Float32Array(B * T * (D1 + D2));
  let o = 0;
  for (let i = 0; i < B * T; i++) {
    const aOff = i * D1;
    const bOff = i * D2;
    for (let j = 0; j < D1; j++) out[o++] = Number(a[aOff + j]);
    for (let j = 0; j < D2; j++) out[o++] = Number(b[bOff + j]);
  }

  return tensorF16OrF32(out, [B, T, D1 + D2]);
}

function makeTimeIds() {
  // [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
  const f = new Float32Array([H, W, 0, 0, H, W]);
  return tensorF16OrF32(f, [1, 6]);
}

export class BrowserSDXLStudent {
  constructor({ te1, te2, unet, vae, scheduler }) {
    this.te1 = te1;
    this.te2 = te2;
    this.unet = unet;
    this.vae = vae;
    this.scheduler = scheduler;
    this.timeIds = makeTimeIds();
  }

  static async create(modelBase = "/models") {
    const sessionOptions = {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "basic",
      enableMemPattern: false,
      enableCpuMemArena: false,
    };

    // IMPORTANT: Text encoders should be FP32 for proper prompt steering
    // FP16 loses precision in embeddings which destroys semantic information
    // TODO: Export text_encoder_fp32.onnx and use it instead of fp16
    console.log("Loading models in parallel...");
    const [te1, te2, unet, vae] = await Promise.all([
      ort.InferenceSession.create(`${modelBase}/text_encoder_fp16.onnx`, sessionOptions),
      ort.InferenceSession.create(`${modelBase}/text_encoder_2_fp16.onnx`, sessionOptions),
      ort.InferenceSession.create(`${modelBase}/unet_student_fp32.onnx`, sessionOptions),
      ort.InferenceSession.create(`${modelBase}/vae_decoder_fp32.onnx`, sessionOptions)
    ]);
    console.log("All models loaded!");

    const cfg = await loadSchedulerConfig(`${modelBase}/scheduler_config.json`);
    const scheduler = new DDIMScheduler(cfg);

    return new BrowserSDXLStudent({ te1, te2, unet, vae, scheduler });
  }

  async encodePrompt(prompt) {
    const { input_ids, attention_mask } = await tokenize(prompt, "/tokenizer");

    console.log(`\n=== TEXT ENCODING DEBUG ===`);
    console.log(`Prompt: "${prompt}"`);
    console.log(`Token IDs (first 15):`, Array.from(input_ids.slice(0, 15)));
    console.log(`Attention mask (first 15):`, Array.from(attention_mask.slice(0, 15)));

    // Count actual tokens (non-padding)
    const numTokens = attention_mask.reduce((sum, val) => sum + (val > 0 ? 1 : 0), 0);
    console.log(`Active tokens: ${numTokens}/${MAX_LEN}`);

    const ids = tensorI64FromInt32Array(input_ids, [1, MAX_LEN]);
    const mask = tensorI64FromInt32Array(attention_mask, [1, MAX_LEN]);

    // Run text_encoder_1
    console.log(`Running text_encoder_1...`);
    const out1 = await this.te1.run({ input_ids: ids, attention_mask: mask });
    // Run text_encoder_2
    console.log(`Running text_encoder_2...`);
    const mask_f32 = tensorF32(Float32Array.from(attention_mask), [1, MAX_LEN]);
    const out2 = await this.te2.run({ input_ids: ids, attention_mask: mask_f32 });

    // You must match output names from your ONNX exports:
    const te1Seq = out1.last_hidden_state;
    const te2Seq = out2.last_hidden_state;

    console.log(`TE1 output: dims=[${te1Seq.dims}], dtype=${te1Seq.type}`);
    console.log(`TE2 output: dims=[${te2Seq.dims}], dtype=${te2Seq.type}`);

    // Check for statistics on TE outputs
    const te1Data = te1Seq.data;
    const te2Data = te2Seq.data;
    let te1Min = Infinity, te1Max = -Infinity, te1Sum = 0;
    let te2Min = Infinity, te2Max = -Infinity, te2Sum = 0;

    for (let i = 0; i < te1Data.length; i++) {
      const v = Number(te1Data[i]);
      te1Min = Math.min(te1Min, v);
      te1Max = Math.max(te1Max, v);
      te1Sum += v;
    }
    for (let i = 0; i < te2Data.length; i++) {
      const v = Number(te2Data[i]);
      te2Min = Math.min(te2Min, v);
      te2Max = Math.max(te2Max, v);
      te2Sum += v;
    }

    const te1Mean = te1Sum / te1Data.length;
    const te2Mean = te2Sum / te2Data.length;

    console.log(`TE1 stats: range=[${te1Min.toFixed(2)}, ${te1Max.toFixed(2)}], mean=${te1Mean.toFixed(3)}`);
    console.log(`TE2 stats: range=[${te2Min.toFixed(2)}, ${te2Max.toFixed(2)}], mean=${te2Mean.toFixed(3)}`);

    const encoder_hidden_states = concatLastDim(te1Seq, te2Seq); // [1,77,2048]
    console.log(`Concatenated dims: [${encoder_hidden_states.dims}]`);
    // Don't clamp text embeddings - we need their full semantic range!

    // Pooled text embeds: from encoder_2 projection output (often "text_embeds")
    const text_embeds = out2.pooled_output ?? out2.text_embeds ?? out2.pooler_output;
    if (!text_embeds) {
      throw new Error("Could not find pooled text embeds output (expected pooled_output, text_embeds, or pooler_output) from text_encoder_2");
    }
    // Don't clamp pooled embeddings either

    // Final embedding statistics
    const encData = encoder_hidden_states.data;
    const poolData = text_embeds.data;
    let encMin = Infinity, encMax = -Infinity, encSum = 0, encZeros = 0;
    let poolMin = Infinity, poolMax = -Infinity, poolSum = 0;

    for (let i = 0; i < encData.length; i++) {
      const v = Number(encData[i]);
      encMin = Math.min(encMin, v);
      encMax = Math.max(encMax, v);
      encSum += v;
      if (Math.abs(v) < 1e-6) encZeros++;
    }
    for (let i = 0; i < poolData.length; i++) {
      const v = Number(poolData[i]);
      poolMin = Math.min(poolMin, v);
      poolMax = Math.max(poolMax, v);
      poolSum += v;
    }

    const encMean = encSum / encData.length;
    const poolMean = poolSum / poolData.length;
    const encZeroPercent = (encZeros / encData.length * 100).toFixed(1);

    console.log(`\nFinal embeddings:`);
    console.log(`  Encoder: range=[${encMin.toFixed(2)}, ${encMax.toFixed(2)}], mean=${encMean.toFixed(3)}, zeros=${encZeroPercent}%`);
    console.log(`  Pooled: range=[${poolMin.toFixed(2)}, ${poolMax.toFixed(2)}], mean=${poolMean.toFixed(3)}, dim=${poolData.length}`);
    console.log(`=== END TEXT ENCODING ===\n`);

    return { encoder_hidden_states, text_embeds };
  }

  async generate(prompt, { steps = 30, seed = 0, guidanceScale = 4.0, negativePrompt = "" } = {}) {
    console.log(`\n=== GENERATION STARTED ===`);
    console.log(`Guidance scale: ${guidanceScale}`);

    // 1) Encode positive prompt
    const { encoder_hidden_states: prompt_embeds, text_embeds: pooled_embeds } = await this.encodePrompt(prompt);

    // 2) Encode negative prompt (for CFG)
    let negative_prompt_embeds, negative_pooled_embeds;
    if (guidanceScale > 1.0) {
      console.log(`\nEncoding negative prompt for CFG...`);
      const neg = await this.encodePrompt(negativePrompt || "");
      negative_prompt_embeds = neg.encoder_hidden_states;
      negative_pooled_embeds = neg.text_embeds;
    }

    // 3) Init latents: Start with clean random latent before adding noise
    const latentSize = 1 * 4 * LAT_H * LAT_W;
    const x0 = randn(latentSize, seed);

    // Generate noise with offset (matches training: noise_offset applied to NOISE, not latent)
    const NOISE_OFFSET = 0.05;
    const noise = randn(latentSize, seed + 1);

    // Apply per-channel offset to NOISE (not x0!)
    for (let c = 0; c < 4; c++) {
      const channelOffset = (Math.random() - 0.5) * 2 * NOISE_OFFSET;
      for (let h = 0; h < LAT_H; h++) {
        for (let w = 0; w < LAT_W; w++) {
          const idx = c * LAT_H * LAT_W + h * LAT_W + w;
          noise[idx] += channelOffset;
        }
      }
    }

    // Create initial noisy latent at t=800: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
    const t_start = 800;
    const sqrt_alpha = Math.sqrt(this.scheduler.alphasCumprod[t_start]);
    const sqrt_one_minus = Math.sqrt(1.0 - this.scheduler.alphasCumprod[t_start]);

    let x = new Float32Array(latentSize);
    for (let i = 0; i < latentSize; i++) {
      x[i] = sqrt_alpha * x0[i] + sqrt_one_minus * noise[i];
    }

    // 4) Timesteps for DDIM (from high noise to low noise)
    // Training range was t=50 to t=800, but we can denoise further toward 0
    // for better final image quality (model can extrapolate somewhat)
    const ts = this.scheduler.timesteps(steps, 10, 800);

    console.log(`Timestep schedule (${steps} steps): [${ts[0]}, ${ts[1]}, ..., ${ts[ts.length - 2]}, ${ts[ts.length - 1]}]`);
    console.log(`Initial latent range: [${Math.min(...x).toFixed(2)}, ${Math.max(...x).toFixed(2)}]`);

    // SDXL VAE scaling factor (applied only before/after VAE operations)
    const LATENT_SCALE = 0.13025;

    // 5) Sampling loop with CFG
    const doCFG = guidanceScale > 1.0;

    for (let i = 0; i < ts.length - 1; i++) {
      const t = ts[i];
      const tPrev = ts[i + 1];

      // Calculate dynamic guidance: stronger in early steps (structure), weaker in late steps (details)
      const progress = i / (ts.length - 1);
      const dynamicGuidance = guidanceScale * (1.0 + 2.0 * (1.0 - progress));

      // For CFG, we need to run UNet twice (unconditional and conditional)
      // instead of batching since the model was exported with batch_size=1
      const latentTensor = tensorF32(x, [1, 4, LAT_H, LAT_W]);

      // Prepare encoder states and embeddings
      const encoderFp32_pos = tensorF32(
        Float32Array.from(prompt_embeds.data, v => Number(v)),
        prompt_embeds.dims
      );
      const embedFp32_pos = tensorF32(
        Float32Array.from(pooled_embeds.data, v => Number(v)),
        pooled_embeds.dims
      );
      const timeFp32 = tensorF32(
        Float32Array.from(this.timeIds.data, v => Number(v)),
        this.timeIds.dims
      );

      let eps_cond;

      // Run conditional prediction
      const out_cond = await this.unet.run({
        latent: latentTensor,
        t: tensorI64FromInt(t),
        encoder_hidden_states: encoderFp32_pos,
        text_embeds: embedFp32_pos,
        time_ids: timeFp32,
      });

      if (!out_cond.noise_pred) {
        throw new Error("UNet did not return noise_pred output");
      }
      eps_cond = toFloat32(out_cond.noise_pred.data);

      let eps;

      if (doCFG) {
        // Run unconditional prediction with negative prompt emphasis
        const encoderFp32_neg = tensorF32(
          Float32Array.from(negative_prompt_embeds.data, v => Number(v) * 1.2),
          negative_prompt_embeds.dims
        );
        const embedFp32_neg = tensorF32(
          Float32Array.from(negative_pooled_embeds.data, v => Number(v) * 1.2),
          negative_pooled_embeds.dims
        );

        const out_uncond = await this.unet.run({
          latent: latentTensor,
          t: tensorI64FromInt(t),
          encoder_hidden_states: encoderFp32_neg,
          text_embeds: embedFp32_neg,
          time_ids: timeFp32,
        });

        const eps_uncond = toFloat32(out_uncond.noise_pred.data);

        // Apply advanced CFG with multiple enhancements:
        // 1. Channel-wise weighting (different guidance per latent channel)
        // 2. Non-linear amplification (exponential scaling of differences)
        // 3. Dynamic timestep-based guidance
        const channelWeights = [1.2, 1.0, 1.0, 0.8]; // Emphasize structure channel
        eps = new Float32Array(eps_cond.length);
        
        for (let k = 0; k < eps.length; k++) {
          const channel = Math.floor((k / (LAT_H * LAT_W))) % 4;
          const weight = channelWeights[channel];
          const diff = eps_cond[k] - eps_uncond[k];
          
          // Non-linear amplification: boost strong signals more than weak ones
          const amplified = diff * (1.0 + 0.1 * Math.abs(diff));
          eps[k] = eps_uncond[k] + dynamicGuidance * weight * amplified;
        }

        // Rescale CFG to prevent oversaturation (normalize magnitude)
        let mag_uncond = 0, mag_cfg = 0;
        for (let k = 0; k < eps.length; k++) {
          mag_uncond += eps_uncond[k] * eps_uncond[k];
          mag_cfg += eps[k] * eps[k];
        }
        mag_uncond = Math.sqrt(mag_uncond / eps.length);
        mag_cfg = Math.sqrt(mag_cfg / eps.length);
        
        const rescale_factor = mag_uncond / (mag_cfg + 1e-8);
        for (let k = 0; k < eps.length; k++) {
          eps[k] *= rescale_factor;
        }

        if (i === 0) {
          console.log(`Advanced CFG: base=${guidanceScale.toFixed(1)}, dynamic=${dynamicGuidance.toFixed(2)}, rescale=${rescale_factor.toFixed(3)}`);
        }
      } else {
        eps = eps_cond;
      }

      // Check for NaN before clamping
      let hasNaN = false;
      for (let k = 0; k < eps.length; k++) {
        if (isNaN(eps[k]) || !isFinite(eps[k])) {
          hasNaN = true;
          break;
        }
      }

      if (hasNaN) {
        console.warn(`Step ${i}: NaN detected in UNet output, replacing with small noise`);
        for (let k = 0; k < eps.length; k++) {
          if (isNaN(eps[k]) || !isFinite(eps[k])) {
            eps[k] = (Math.random() - 0.5) * 0.001;
          }
        }
      }

      // Very light clamping - only prevent extreme outliers
      clampArray(eps, -10000, 10000);

      // Debug first and last steps
      if (i === 0 || i === ts.length - 2) {
        const epsMin = Math.min(...eps);
        const epsMax = Math.max(...eps);
        console.log(`Step ${i}/${ts.length - 1} t=${t}->${tPrev}: eps range [${epsMin.toFixed(2)}, ${epsMax.toFixed(2)}]`);
      }

      // DDIM update
      x = this.scheduler.step({ x, eps, t, tPrev });

      // Debug latent range
      if (i === 0 || i === ts.length - 2) {
        const xMin = Math.min(...x);
        const xMax = Math.max(...x);
        console.log(`  -> latent range [${xMin.toFixed(2)}, ${xMax.toFixed(2)}]`);
      }

      // Light clamping only to prevent divergence
      clampArray(x, -1000, 1000);
    }

    // 5) VAE decode
    // Unscale latents before VAE (SDXL VAE expects unscaled latents)
    const unscaled = new Float32Array(x.length);
    for (let k = 0; k < x.length; k++) unscaled[k] = x[k] / LATENT_SCALE;

    // Debug final latent stats
    const latMin = Math.min(...unscaled);
    const latMax = Math.max(...unscaled);
    console.log(`Final unscaled latents: [${latMin.toFixed(2)}, ${latMax.toFixed(2)}]`);

    // Reasonable clamping for VAE stability
    clampArray(unscaled, -100, 100);

    const latentsTensor = tensorF32(unscaled, [1, 4, LAT_H, LAT_W]);

    const imgOut = await this.vae.run({ latents: latentsTensor });

    return imgOut.image; // [1,3,256,512] float
  }
}

function toFloat32(arr) {
  if (arr instanceof Float32Array) return arr;
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = Number(arr[i]);
  return out;
}

export function renderCHW01ToCanvas(imageTensor, canvas) {
  const [B, C, Hh, Ww] = imageTensor.dims;
  const data = imageTensor.data;

  const ctx = canvas.getContext("2d");
  canvas.width = Ww;
  canvas.height = Hh;

  const rgba = new Uint8ClampedArray(Ww * Hh * 4);
  for (let y = 0; y < Hh; y++) {
    for (let x = 0; x < Ww; x++) {
      const idx = y * Ww + x;
      const r = Number(data[(0 * Hh + y) * Ww + x]);
      const g = Number(data[(1 * Hh + y) * Ww + x]);
      const b = Number(data[(2 * Hh + y) * Ww + x]);
      // VAE outputs are in [-1, 1] range, convert to [0, 255]
      rgba[idx * 4 + 0] = Math.max(0, Math.min(255, ((r + 1) * 127.5) | 0));
      rgba[idx * 4 + 1] = Math.max(0, Math.min(255, ((g + 1) * 127.5) | 0));
      rgba[idx * 4 + 2] = Math.max(0, Math.min(255, ((b + 1) * 127.5) | 0));
      rgba[idx * 4 + 3] = 255;
    }
  }

  ctx.putImageData(new ImageData(rgba, Ww, Hh), 0, 0);
}
