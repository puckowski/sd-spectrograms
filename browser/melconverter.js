// Mel Spectrogram to Audio Converter
// Integrated from mel.html for browser-based audio reconstruction

// ------------------------
// Minimal FFT (radix-2) complex FFT
// ------------------------
class FFT {
  constructor(n) {
    if ((n & (n - 1)) !== 0) throw new Error("n_fft must be power of 2");
    this.n = n;
    this._cos = new Float32Array(n / 2);
    this._sin = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      const ang = -2 * Math.PI * i / n;
      this._cos[i] = Math.cos(ang);
      this._sin[i] = Math.sin(ang);
    }
  }

  forward(re, im) {
    const n = this.n;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const idx = k * step;
          const c = this._cos[idx];
          const s = this._sin[idx];
          const tre = re[i + k + half] * c - im[i + k + half] * s;
          const tim = re[i + k + half] * s + im[i + k + half] * c;
          re[i + k + half] = re[i + k] - tre;
          im[i + k + half] = im[i + k] - tim;
          re[i + k] += tre;
          im[i + k] += tim;
        }
      }
    }
  }

  inverse(re, im) {
    const n = this.n;
    for (let i = 0; i < n; i++) im[i] = -im[i];
    this.forward(re, im);
    for (let i = 0; i < n; i++) {
      re[i] = re[i] / n;
      im[i] = -im[i] / n;
    }
  }
}

// ------------------------
// Windows & helpers
// ------------------------
function hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
  return w;
}

function clip(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// ------------------------
// Mel filterbank (triangular)
// ------------------------
function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1); }
function linspace(a, b, n) {
  const out = new Float32Array(n);
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = a + step * i;
  return out;
}

function createMelFilterbank({ sr, nFft, nMels, fmin, fmax }) {
  const nFreq = (nFft / 2) + 1;
  const melMin = hzToMel(fmin);
  const melMax = hzToMel(fmax);
  const melPts = linspace(melMin, melMax, nMels + 2);
  const hzPts = Array.from(melPts, melToHz);
  const bins = hzPts.map(hz => Math.floor((nFft + 1) * hz / sr));
  const fb = Array.from({ length: nMels }, () => new Float32Array(nFreq));

  for (let m = 1; m <= nMels; m++) {
    const left = bins[m - 1], center = bins[m], right = bins[m + 1];
    for (let k = left; k < center; k++) if (k >= 0 && k < nFreq) fb[m - 1][k] = (k - left) / Math.max(1, (center - left));
    for (let k = center; k < right; k++) if (k >= 0 && k < nFreq) fb[m - 1][k] = (right - k) / Math.max(1, (right - center));
  }
  return fb;
}

// Approximate mel (power) -> linear (power) using transpose weighting (pseudo-inverse-ish)
function melToLinearApprox(melPower, melFilterbank) {
  const nMels = melPower.length;
  const nFrames = melPower[0].length;
  const nFreq = melFilterbank[0].length;

  const denom = new Float32Array(nFreq);
  for (let m = 0; m < nMels; m++) {
    const f = melFilterbank[m];
    for (let k = 0; k < nFreq; k++) denom[k] += f[k] * f[k];
  }
  for (let k = 0; k < nFreq; k++) denom[k] = Math.max(1e-8, denom[k]);

  const lin = Array.from({ length: nFreq }, () => new Float32Array(nFrames));

  for (let m = 0; m < nMels; m++) {
    const f = melFilterbank[m];
    const mp = melPower[m];
    for (let k = 0; k < nFreq; k++) {
      const w = f[k];
      if (w === 0) continue;
      const w2 = w / denom[k];
      for (let t = 0; t < nFrames; t++) lin[k][t] += mp[t] * w2;
    }
  }
  return lin;
}

function powerToMagnitude(powerSpec) {
  const nFreq = powerSpec.length;
  const nFrames = powerSpec[0].length;
  const mag = Array.from({ length: nFreq }, () => new Float32Array(nFrames));
  for (let k = 0; k < nFreq; k++) {
    const p = powerSpec[k];
    const o = mag[k];
    for (let t = 0; t < nFrames; t++) o[t] = Math.sqrt(Math.max(0, p[t]));
  }
  return mag;
}

// ------------------------
// Griffin–Lim (center=True-ish padding is ignored; we just do OLA)
// ------------------------
function griffinLim({ mag, nFft, hop, nIter }) {
  const nFreq = mag.length;
  const nFrames = mag[0].length;
  const fft = new FFT(nFft);
  const window = hann(nFft);

  const phase = Array.from({ length: nFrames }, () => new Float32Array(nFreq));
  for (let t = 0; t < nFrames; t++) for (let k = 0; k < nFreq; k++) phase[t][k] = (Math.random() * 2 - 1) * Math.PI;

  const outLen = (nFrames - 1) * hop + nFft;
  let y = new Float32Array(outLen);

  for (let iter = 0; iter < nIter; iter++) {
    y.fill(0);

    // ISTFT
    for (let t = 0; t < nFrames; t++) {
      const re = new Float32Array(nFft);
      const im = new Float32Array(nFft);

      for (let k = 0; k < nFreq; k++) {
        const a = mag[k][t];
        const p = phase[t][k];
        re[k] = a * Math.cos(p);
        im[k] = a * Math.sin(p);
      }
      for (let k = 1; k < nFreq - 1; k++) {
        re[nFft - k] = re[k];
        im[nFft - k] = -im[k];
      }

      fft.inverse(re, im);

      const offset = t * hop;
      for (let i = 0; i < nFft; i++) y[offset + i] += re[i] * window[i];
    }

    // Recompute phase
    for (let t = 0; t < nFrames; t++) {
      const offset = t * hop;
      const frame = new Float32Array(nFft);
      for (let i = 0; i < nFft; i++) frame[i] = (y[offset + i] ?? 0) * window[i];

      const re = new Float32Array(nFft);
      const im = new Float32Array(nFft);
      re.set(frame);
      fft.forward(re, im);

      for (let k = 0; k < nFreq; k++) phase[t][k] = Math.atan2(im[k], re[k]);
    }
  }

  return y;
}

// ------------------------
// Pre/De-emphasis
// ------------------------
function deemphasis(x, coef = 0.97) {
  // Matches librosa.effects.deemphasis recurrence: y[n] = x[n] + coef*y[n-1]
  const y = new Float32Array(x.length);
  if (x.length === 0) return y;
  y[0] = x[0];
  for (let n = 1; n < x.length; n++) y[n] = x[n] + coef * y[n - 1];
  return y;
}

// ------------------------
// LUFS-ish normalize + true-peak-ish limit
// Browser does NOT implement EBU R128 gating/filtering here.
// This is an approximation so your "settings" do something.
// ------------------------
function rms(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, x.length));
}

// Approximate LUFS from RMS. Calibrated so sine @ -18 dBFS ~ -18 LUFS-ish.
// Treat as "good enough for leveling", not compliance.
function approxIntegratedLufs(x) {
  const r = rms(x);
  const dbfs = 20 * Math.log10(r + 1e-12);
  return dbfs; // rough
}

function oversampledPeak(x, oversample = 2) {
  if (oversample <= 1) {
    let p = 0;
    for (let i = 0; i < x.length; i++) p = Math.max(p, Math.abs(x[i]));
    return p;
  }
  let p = 0;
  // linear interpolation oversampling (like your python quick TP check)
  for (let i = 0; i < x.length - 1; i++) {
    const a = x[i], b = x[i + 1];
    for (let j = 0; j < oversample; j++) {
      const t = j / oversample;
      const v = a + (b - a) * t;
      p = Math.max(p, Math.abs(v));
    }
  }
  p = Math.max(p, Math.abs(x[x.length - 1]));
  return p;
}

function lufsNormalizeWithLimiting(x, targetLufs = -14.0, tpLimitDb = -1.0, oversample = 2) {
  const loudness = approxIntegratedLufs(x);
  const gainDb = targetLufs - loudness;
  const g = Math.pow(10, gainDb / 20);

  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i] * g;

  const peak = oversampledPeak(y, oversample);
  const limitLin = Math.pow(10, tpLimitDb / 20);
  if (peak > limitLin) {
    const s = limitLin / (peak + 1e-12);
    for (let i = 0; i < y.length; i++) y[i] *= s;
  }
  return { y, loudness, gainDb };
}

// ------------------------
// Canvas -> grayscale [0,1] matrix
// ------------------------
function canvasToGrayMatrix(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);

  let mat = Array.from({ length: height }, () => new Float32Array(width));
  for (let y = 0; y < height; y++) {
    const row = mat[y];
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      row[x] = data[idx] / 255; // red channel
    }
  }

  // If saved transposed (512x256), transpose back to (256x512)
  if (height === 512 && width === 256) {
    const trans = Array.from({ length: 256 }, () => new Float32Array(512));
    for (let y = 0; y < 512; y++) {
      for (let x = 0; x < 256; x++) trans[x][y] = mat[y][x];
    }
    mat = trans;
  }

  return mat; // [N_MELS][T] expected -> [256][512]
}

// ------------------------
// Denorm dB & to mel power (match python)
// ------------------------
function dbImgToMelPower(img01, dbMin, dbMax) {
  const nMels = img01.length;
  const nFrames = img01[0].length;
  const melPower = Array.from({ length: nMels }, () => new Float32Array(nFrames));

  for (let m = 0; m < nMels; m++) {
    const row = img01[m];
    const out = melPower[m];
    for (let t = 0; t < nFrames; t++) {
      const sNorm = clip(row[t], 0, 1);
      const db = sNorm * (dbMax - dbMin) + dbMin;
      out[t] = Math.pow(10, db / 10); // db_to_power
    }
  }
  return melPower;
}

// ------------------------
// WAV encoding + playback
// ------------------------
function floatToWavBlob(samples, sampleRate) {
  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");

  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);

  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = clip(samples[i], -1, 1);
    view.setInt16(o, Math.round(s * 32767), true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

async function playFloat32(samples, sampleRate) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
  const buf = ctx.createBuffer(1, samples.length, sampleRate);
  buf.getChannelData(0).set(samples);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  return { ctx, src };
}

// ------------------------
// Main conversion function
// ------------------------
async function convertCanvasToAudio() {
  const canvas = document.getElementById("out");
  const statusEl = document.getElementById("audioStatus");
  const outputEl = document.getElementById("audioOutput");

  if (!canvas.width || !canvas.height) {
    statusEl.textContent = "Error: No image generated yet. Please generate an image first.";
    return;
  }

  const c = {
    sr: +document.getElementById("sr").value,
    nMels: +document.getElementById("nmels").value,
    dur: +document.getElementById("dur").value,
    nFft: +document.getElementById("nfft").value,
    hop: +document.getElementById("hop").value,
    iters: +document.getElementById("iters").value,
    dbmin: +document.getElementById("dbmin").value,
    dbmax: +document.getElementById("dbmax").value,
    fmin: +document.getElementById("fmin").value,
    fmax: +document.getElementById("fmax").value,
    preemph: +document.getElementById("preemph").value,
    deemph: document.getElementById("deemph").checked,
    applyLufs: document.getElementById("applyLufs").checked,
    targetLufs: +document.getElementById("targetLufs").value,
    tpLimitDb: +document.getElementById("tpLimitDb").value,
    oversample: +document.getElementById("oversample").value
  };

  try {
    statusEl.textContent = "Building mel filterbank...";
    const melFB = createMelFilterbank({ sr: c.sr, nFft: c.nFft, nMels: c.nMels, fmin: c.fmin, fmax: c.fmax });

    statusEl.textContent = "Loading image from canvas...";
    const img = canvasToGrayMatrix(canvas);

    const H = img.length, W = img[0]?.length ?? 0;
    if (!(H === c.nMels && W === 512)) {
      statusEl.textContent = `Error: unexpected shape ${H}x${W} (expected ${c.nMels}x512)`;
      return;
    }

    statusEl.textContent = `Reconstructing audio... (GL iters=${c.iters})`;

    // 1) Denorm dB -> mel power
    const melPower = dbImgToMelPower(img, c.dbmin, c.dbmax);

    // 2) Approx mel -> linear (power)
    const linPower = melToLinearApprox(melPower, melFB);

    // 3) Power -> magnitude
    const linMag = powerToMagnitude(linPower);

    // 4) Griffin–Lim
    let y = griffinLim({ mag: linMag, nFft: c.nFft, hop: c.hop, nIter: c.iters });

    // 5) De-emphasis (invert pre-emphasis from forward pass)
    if (c.deemph) y = deemphasis(y, c.preemph);

    // 6) Optional LUFS normalize + TP limit (approx)
    let lufsIn = null, gainDb = 0;
    if (c.applyLufs) {
      const r = lufsNormalizeWithLimiting(y, c.targetLufs, c.tpLimitDb, c.oversample);
      y = r.y;
      lufsIn = r.loudness;
      gainDb = r.gainDb;
    }

    // 7) Force duration (trim/pad)
    const need = Math.max(1, Math.round(c.dur * c.sr));
    if (y.length > need) y = y.slice(0, need);
    else if (y.length < need) {
      const z = new Float32Array(need);
      z.set(y);
      y = z;
    }

    // Small final peak safety
    let peak = 1e-9;
    for (let k = 0; k < y.length; k++) peak = Math.max(peak, Math.abs(y[k]));
    const scale = peak > 0.98 ? 0.98 / peak : 1.0;
    if (scale !== 1.0) for (let k = 0; k < y.length; k++) y[k] *= scale;

    // UI output
    outputEl.innerHTML = "";
    const card = document.createElement("div");
    card.className = "card";

    const title = document.createElement("div");
    title.innerHTML = `<b>Generated Audio</b>`;

    const row = document.createElement("div");
    row.className = "row";

    const playBtn = document.createElement("button");
    playBtn.textContent = "Play";
    playBtn.onclick = () => playFloat32(y, c.sr);

    const wavBlob = floatToWavBlob(y, c.sr);
    const a = document.createElement("a");
    a.textContent = "Download WAV";
    a.href = URL.createObjectURL(wavBlob);
    a.download = "mel_spectrogram_recon.wav";
    a.style.marginLeft = "10px";

    const meta = document.createElement("div");
    meta.className = "small";
    meta.style.marginTop = "8px";
    meta.textContent =
      `Samples: ${y.length} • SR: ${c.sr}` +
      (lufsIn === null ? "" : ` • LUFS_in≈${lufsIn.toFixed(2)} • gain_db≈${gainDb.toFixed(2)}`);

    row.appendChild(playBtn);
    row.appendChild(a);

    card.appendChild(title);
    card.appendChild(row);
    card.appendChild(meta);
    outputEl.appendChild(card);

    statusEl.textContent = "Done!";

  } catch (e) {
    statusEl.textContent = `Error: ${e?.message ?? e}`;
    console.error(e);
  }
}

// ------------------------
// UI Setup
// ------------------------
window.addEventListener("DOMContentLoaded", () => {
  const convertBtn = document.getElementById("convertBtn");
  if (convertBtn) {
    convertBtn.onclick = convertCanvasToAudio;
  }

  // Show audio controls when canvas has content
  const canvas = document.getElementById("out");
  const observer = new MutationObserver(() => {
    if (canvas.width > 0 && canvas.height > 0) {
      document.getElementById("audioControls").style.display = "block";
    }
  });

  // Watch for canvas changes
  const checkCanvas = setInterval(() => {
    if (canvas.width > 0 && canvas.height > 0) {
      document.getElementById("audioControls").style.display = "block";
      clearInterval(checkCanvas);
    }
  }, 500);
});
