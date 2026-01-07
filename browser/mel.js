import fs from "fs-extra";
import path from "path";
import glob from "glob";
import { PNG } from "pngjs";
import FFT from "fft.js";
import wavEncoder from "wav-encoder";

// ---------------- CONFIG ----------------
const SR = 16000;
const N_MELS = 256;
const DURATION = 15;
const DB_MIN = -80;
const DB_MAX = 0;
const N_FFT = 2048;
const HOP_LENGTH = 512;
const N_ITER = 32;

const INPUT_GLOB = "spectrograms/20_song/*.png";
const OUTPUT_DIR = "reconstructed_audio";

await fs.ensureDir(OUTPUT_DIR);

// ---------------- UTILS ----------------

function dbToPower(db) {
  return Math.pow(10, db / 10);
}

function linspace(start, end, n) {
  return Array.from({ length: n }, (_, i) =>
    start + (i / (n - 1)) * (end - start)
  );
}

// ---------------- MEL FILTERBANK ----------------
// Simplified mel filterbank (same logic as librosa)
function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

function createMelFilterbank() {
  const melMin = hzToMel(0);
  const melMax = hzToMel(SR / 2);
  const melPoints = linspace(melMin, melMax, N_MELS + 2);
  const hzPoints = melPoints.map(melToHz);
  const bin = hzPoints.map(hz =>
    Math.floor((N_FFT + 1) * hz / SR)
  );

  const filters = Array.from({ length: N_MELS }, () =>
    new Float32Array(N_FFT / 2 + 1)
  );

  for (let m = 1; m <= N_MELS; m++) {
    for (let k = bin[m - 1]; k < bin[m]; k++) {
      filters[m - 1][k] =
        (k - bin[m - 1]) / (bin[m] - bin[m - 1]);
    }
    for (let k = bin[m]; k < bin[m + 1]; k++) {
      filters[m - 1][k] =
        (bin[m + 1] - k) / (bin[m + 1] - bin[m]);
    }
  }

  return filters;
}

const MEL_FILTERBANK = createMelFilterbank();

// ---------------- GRIFFIN–LIM ----------------
function griffinLim(magSpec) {
  const fft = new FFT(N_FFT);
  const frames = magSpec[0].length;

  let phase = Array.from({ length: frames }, () =>
    Float32Array.from({ length: N_FFT / 2 + 1 }, () =>
      Math.random() * 2 * Math.PI
    )
  );

  let signal = new Float32Array(frames * HOP_LENGTH + N_FFT);

  for (let iter = 0; iter < N_ITER; iter++) {
    signal.fill(0);

    for (let t = 0; t < frames; t++) {
      const re = new Float32Array(N_FFT);
      const im = new Float32Array(N_FFT);

      for (let k = 0; k < magSpec.length; k++) {
        re[k] = magSpec[k][t] * Math.cos(phase[t][k]);
        im[k] = magSpec[k][t] * Math.sin(phase[t][k]);
      }

      fft.inverseTransform(re, im);
      const frame = fft.fromComplexArray(re, im);

      for (let i = 0; i < N_FFT; i++) {
        signal[t * HOP_LENGTH + i] += frame[i];
      }
    }

    // Recompute phase
    for (let t = 0; t < frames; t++) {
      const frame = signal.slice(
        t * HOP_LENGTH,
        t * HOP_LENGTH + N_FFT
      );

      const re = new Float32Array(N_FFT);
      const im = new Float32Array(N_FFT);
      fft.realTransform(re, frame);
      fft.completeSpectrum(re, im);

      for (let k = 0; k < phase[t].length; k++) {
        phase[t][k] = Math.atan2(im[k], re[k]);
      }
    }
  }

  return signal;
}

// ---------------- MAIN LOOP ----------------

for (const file of glob.sync(INPUT_GLOB)) {
  const png = PNG.sync.read(await fs.readFile(file));
  const { width, height, data } = png;

  // Extract grayscale
  let img = [];
  for (let y = 0; y < height; y++) {
    const row = [];
    for (let x = 0; x < width; x++) {
      row.push(data[(y * width + x) * 4] / 255);
    }
    img.push(row);
  }

  // Fix orientation
  if (height === 472 && width === 256) {
    img = img[0].map((_, i) => img.map(r => r[i]));
  }

  if (img.length !== 256 || img[0].length !== 472) {
    console.warn(`Skipping ${file}, bad shape`);
    continue;
  }

  // dB → power
  const melSpec = img.map(row =>
    row.map(v => dbToPower(v * (DB_MAX - DB_MIN) + DB_MIN))
  );

  // Approximate mel → linear
  const linearSpec = MEL_FILTERBANK.map((filt, m) =>
    melSpec[m]
  );

  // Griffin–Lim
  let audio = griffinLim(linearSpec);
  audio = audio.slice(0, SR * DURATION);

  const outPath = path.join(
    OUTPUT_DIR,
    path.basename(file, ".png") + "_reconstructed.wav"
  );

  await wavEncoder.writeFile(outPath, {
    sampleRate: SR,
    getChannelData: () => audio
  });

  console.log(`Wrote: ${outPath}`);
}
