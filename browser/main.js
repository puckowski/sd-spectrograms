import { BrowserSDXLStudent, renderCHW01ToCanvas } from "./sd_pipeline.js";

let model = null;

async function ensureModel() {
  if (!model) {
    model = await BrowserSDXLStudent.create("/models");
  }
  return model;
}

window.runDemo = async function runDemo() {
  const prompt = document.getElementById("prompt").value;
  const canvas = document.getElementById("out");
  const status = document.getElementById("status");

  status.textContent = "Loading model...";
  const m = await ensureModel();

  status.textContent = "Generating with CFG...";
  const img = await m.generate(prompt, { 
    steps: 30, 
    seed: 1234,
    guidanceScale: 4.0,  // Match Python inference
    negativePrompt: ""   // Optional negative prompt
  });

  status.textContent = "Rendering...";
  renderCHW01ToCanvas(img, canvas);

  status.textContent = "Done.";
};
