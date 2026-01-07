// DDIM scheduler for epsilon-prediction.
// Reads betas from scheduler_config.json and builds alphas_cumprod.
//
// This is a "correct" sampler baseline. You can later replace with Euler/Heun/DPM++.

export class DDIMScheduler {
  constructor(cfg) {
    this.cfg = cfg;

    const betas = cfg.betas;
    if (!Array.isArray(betas) || betas.length === 0) {
      throw new Error("scheduler_config.json must contain an array 'betas'");
    }

    this.numTrainTimesteps = betas.length;
    this.betas = new Float32Array(betas);
    this.alphas = new Float32Array(betas.length);
    this.alphasCumprod = new Float32Array(betas.length);

    let cum = 1.0;
    for (let i = 0; i < betas.length; i++) {
      const a = 1.0 - this.betas[i];
      this.alphas[i] = a;
      cum *= a;
      this.alphasCumprod[i] = cum;
    }

    this.eta = 0.0; // 0.0 => deterministic DDIM
  }

  /**
   * Create a timestep schedule matching training range.
   * Training used t_lo=50 to t_hi=800, so inference must stay in that range.
   * steps: inference steps, e.g. 30
   */
  timesteps(steps, t_lo = 50, t_hi = 800) {
    const out = new Int32Array(steps);
    for (let i = 0; i < steps; i++) {
      const frac = i / (steps - 1);
      // Linear interpolation from t_hi down to t_lo
      out[i] = Math.round(t_hi - (t_hi - t_lo) * frac);
    }
    return out;
  }

  /**
   * DDIM step: x_t -> x_{t_prev}
   * Inputs are Float32Array for latents and eps.
   *
   * Standard DDIM formula (eta=0, deterministic):
   * pred_x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
   * dir_xt = sqrt(1 - alpha_prev) * eps
   * x_{t-1} = sqrt(alpha_prev) * pred_x0 + dir_xt
   */
  step({ x, eps, t, tPrev }) {
    const a_t = this.alphasCumprod[t];
    const a_prev = this.alphasCumprod[tPrev];

    const sqrt_a_t = Math.sqrt(a_t);
    const sqrt_one_t = Math.sqrt(1 - a_t);
    const sqrt_a_prev = Math.sqrt(a_prev);
    const sqrt_one_prev = Math.sqrt(1 - a_prev);

    // Predict x0 from xt and epsilon
    const pred_x0 = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      pred_x0[i] = (x[i] - sqrt_one_t * eps[i]) / sqrt_a_t;
    }

    // Compute x_{t-1}
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      out[i] = sqrt_a_prev * pred_x0[i] + sqrt_one_prev * eps[i];
    }

    return out;
  }

}

// Load scheduler config from public path
export async function loadSchedulerConfig(path = "/models/scheduler_config.json") {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Failed to load scheduler config: ${path}`);
  return await r.json();
}
