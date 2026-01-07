import { AutoTokenizer, env } from "./transformers.min.js";

const MAX_LEN = 77;

env.allowLocalModels = true;
env.allowRemoteModels = false;

let _tokenizerPromise = null;

export async function loadTokenizer(tokenizerPath = "/tokenizer") {
  if (!_tokenizerPromise) _tokenizerPromise = AutoTokenizer.from_pretrained(tokenizerPath);
  return _tokenizerPromise;
}

export async function tokenize(prompt, tokenizerPath = "/tokenizer") {
  const tok = await loadTokenizer(tokenizerPath);

  const enc = await tok(prompt, {
    padding: "max_length",
    truncation: true,
    max_length: MAX_LEN,
    return_attention_mask: true,
  });

  // DEBUG (optional): uncomment once to see what Transformers.js is returning
  // console.log("enc.input_ids=", enc.input_ids);
  // console.log("enc.attention_mask=", enc.attention_mask);

  const inputIds = await toInt32Fixed(enc.input_ids, MAX_LEN);
  const attnMask = await toInt32Fixed(enc.attention_mask, MAX_LEN);

  return { input_ids: inputIds, attention_mask: attnMask };
}

async function toInt32Fixed(x, len) {
  const out = new Int32Array(len);
  if (x == null) return out;

  // 1) If x is a Transformers.js Tensor-like object with .data
  // (common: { data: TypedArray, dims: [...] } )
  if (typeof x === "object" && x.data && ArrayBuffer.isView(x.data)) {
    const a = x.data;
    const n = Math.min(len, a.length);
    for (let i = 0; i < n; i++) out[i] = Number(a[i]);
    return out;
  }

  // 2) If x has tolist() (some tensor wrappers)
  if (typeof x === "object" && typeof x.tolist === "function") {
    const arr = await x.tolist(); // may be nested
    return toInt32Fixed(arr, len);
  }

  // 3) If x is already a TypedArray
  if (ArrayBuffer.isView(x)) {
    const n = Math.min(len, x.length);
    for (let i = 0; i < n; i++) out[i] = Number(x[i]);
    return out;
  }

  // 4) Nested array [[...]]
  if (Array.isArray(x) && Array.isArray(x[0])) {
    const row = x[0];
    const n = Math.min(len, row.length);
    for (let i = 0; i < n; i++) out[i] = Number(row[i]);
    return out;
  }

  // 5) Flat array [...]
  if (Array.isArray(x)) {
    const n = Math.min(len, x.length);
    for (let i = 0; i < n; i++) out[i] = Number(x[i]);
    return out;
  }

  // 6) Some versions may return { ort_tensor: { data: ... } }
  if (typeof x === "object" && x.ort_tensor && x.ort_tensor.data && ArrayBuffer.isView(x.ort_tensor.data)) {
    const a = x.ort_tensor.data;
    const n = Math.min(len, a.length);
    for (let i = 0; i < n; i++) out[i] = Number(a[i]);
    return out;
  }

  throw new Error(
    `Unsupported tokenizer output type for ids/mask: ${Object.prototype.toString.call(x)} keys=${typeof x === "object" ? Object.keys(x).join(",") : ""}`
  );
}
