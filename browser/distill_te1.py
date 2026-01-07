#!/usr/bin/env python3
"""
Distill SDXL TE1 (text_encoder_1) from caption .txt files WITHOUT loading the full SDXL pipeline,
with metrics every N steps, and correct AMP behavior (student weights FP32; autocast for fp16/bf16).

Teacher:
  --merged_model_dir (diffusers-style folder) subfolders: text_encoder/, tokenizer/

Student:
  initialized from teacher weights but kept in FP32 to avoid "Attempting to unscale FP16 gradients"

Captions:
  recursively finds *.txt in --images_dir

Outputs:
  --output_dir/text_encoder/   (student TE1)
  --output_dir/tokenizer/      (tokenizer_1)
  --output_dir/distill_config.json
"""

# ----------------------------
# QUIET MODE: must be first
# ----------------------------
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import warnings
warnings.filterwarnings(
    "ignore",
    message="`torch.utils._pytree._register_pytree_node` is deprecated.*",
    category=FutureWarning,
)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import CLIPTokenizer, CLIPTextModel


# ----------------------------
# Dataset
# ----------------------------
def _list_caption_files(images_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


class CaptionTxtDataset(Dataset):
    def __init__(
        self,
        caption_files: List[str],
        empty_caption_p: float = 0.0,
        keep_newlines: bool = False,
        shuffle_caption_lines: bool = False,
        max_chars: int = 4000,
    ):
        self.caption_files = caption_files
        self.empty_caption_p = float(empty_caption_p)
        self.keep_newlines = bool(keep_newlines)
        self.shuffle_caption_lines = bool(shuffle_caption_lines)
        self.max_chars = int(max_chars)

    def __len__(self) -> int:
        return len(self.caption_files)

    def _read_caption(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        txt = txt[: self.max_chars]

        if not self.keep_newlines:
            txt = " ".join([line.strip() for line in txt.splitlines() if line.strip()])
        else:
            txt = txt.strip()

        if self.shuffle_caption_lines:
            parts = [x.strip() for x in txt.split(",") if x.strip()]
            random.shuffle(parts)
            txt = ", ".join(parts)

        if random.random() < self.empty_caption_p:
            return ""

        return txt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.caption_files[idx]
        return {"caption": self._read_caption(p), "path": p}


@dataclass
class CollateCaptions:
    tokenizer: CLIPTokenizer
    max_length: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        captions = [x["caption"] for x in batch]
        paths = [x["path"] for x in batch]
        toks = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"input_ids": toks.input_ids, "attention_mask": toks.attention_mask, "paths": paths}


# ----------------------------
# Metrics + Loss helpers
# ----------------------------
def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b, reduction="mean")


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float() ** 2) + eps)


def cosine_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.float()
    b = b.float()
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1).mean()


def grad_global_norm(model: torch.nn.Module) -> float:
    g2 = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g2 += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(g2)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_model_dir", type=str, required=True)
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--train_steps", type=int, default=100000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--w_token", type=float, default=1.0)
    ap.add_argument("--w_pooled", type=float, default=1.0)

    ap.add_argument("--empty_caption_p", type=float, default=0.0)
    ap.add_argument("--shuffle_caption_lines", action="store_true")
    ap.add_argument("--keep_newlines", action="store_true")

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--metrics_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=2000)

    ap.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--num_workers", type=int, default=2)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)
    device = accelerator.device

    # Teacher can be lower precision (frozen). Student MUST be FP32 to avoid FP16-grad unscale error.
    teacher_dtype = torch.float32
    if args.mixed_precision == "fp16":
        teacher_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        teacher_dtype = torch.bfloat16

    if accelerator.is_main_process:
        print(f"[load] tokenizer + TE1 from: {args.merged_model_dir}")

    tokenizer_1 = CLIPTokenizer.from_pretrained(args.merged_model_dir, subfolder="tokenizer")

    teacher_te1 = CLIPTextModel.from_pretrained(
        args.merged_model_dir,
        subfolder="text_encoder",
        torch_dtype=teacher_dtype,   # ok: frozen
    )

    # IMPORTANT: student stays FP32 regardless of mixed_precision
    student_te1 = CLIPTextModel.from_pretrained(
        args.merged_model_dir,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
    )

    teacher_te1.eval()
    for p in teacher_te1.parameters():
        p.requires_grad_(False)

    student_te1.train()
    for p in student_te1.parameters():
        p.requires_grad_(True)

    caption_files = _list_caption_files(args.images_dir)
    if len(caption_files) == 0:
        raise RuntimeError(f"No .txt caption files found under: {args.images_dir}")
    if accelerator.is_main_process:
        print(f"[data] found {len(caption_files)} caption files")

    max_len = int(getattr(tokenizer_1, "model_max_length", 77) or 77)

    ds = CaptionTxtDataset(
        caption_files,
        empty_caption_p=args.empty_caption_p,
        keep_newlines=args.keep_newlines,
        shuffle_caption_lines=args.shuffle_caption_lines,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=CollateCaptions(tokenizer=tokenizer_1, max_length=max_len),
    )

    optim = torch.optim.AdamW(
        student_te1.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.train_steps - args.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    student_te1, teacher_te1, optim, dl, sched = accelerator.prepare(
        student_te1, teacher_te1, optim, dl, sched
    )

    global_step = 0
    t0 = time.time()
    dl_iter = iter(dl)
    eps = 1e-8

    while global_step < args.train_steps:
        student_te1.train()
        optim.zero_grad(set_to_none=True)

        total_loss = 0.0

        metric_token_rel = 0.0
        metric_pooled_rel = 0.0
        metric_pooled_cos = 0.0
        metric_token_mse = 0.0
        metric_pooled_mse = 0.0
        metric_count = 0

        for _ in range(args.grad_accum):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)

            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            # autocast affects compute, not storage; student params remain fp32 => grads fp32
            with accelerator.autocast():
                with torch.no_grad():
                    t = teacher_te1(input_ids=input_ids, attention_mask=attn, return_dict=True)
                    t_token = t.last_hidden_state
                    t_pooled = t.pooler_output

                s = student_te1(input_ids=input_ids, attention_mask=attn, return_dict=True)
                s_token = s.last_hidden_state
                s_pooled = s.pooler_output

                loss_token = mse(s_token.float(), t_token.float())
                loss_pooled = mse(s_pooled.float(), t_pooled.float())
                loss = args.w_token * loss_token + args.w_pooled * loss_pooled

            accelerator.backward(loss / args.grad_accum)
            total_loss += float(loss.detach().item())

            with torch.no_grad():
                token_mse_v = float(loss_token.detach().item())
                pooled_mse_v = float(loss_pooled.detach().item())

                token_rmse = math.sqrt(max(token_mse_v, 0.0))
                pooled_rmse = math.sqrt(max(pooled_mse_v, 0.0))

                token_rel = float(token_rmse / (float(rms(t_token).detach().item()) + eps))
                pooled_rel = float(pooled_rmse / (float(rms(t_pooled).detach().item()) + eps))

                pooled_cos = float(cosine_mean(s_pooled, t_pooled).detach().item())

            metric_token_mse += token_mse_v
            metric_pooled_mse += pooled_mse_v
            metric_token_rel += token_rel
            metric_pooled_rel += pooled_rel
            metric_pooled_cos += pooled_cos
            metric_count += 1

        denom = max(1, metric_count)
        metric_token_mse /= denom
        metric_pooled_mse /= denom
        metric_token_rel /= denom
        metric_pooled_rel /= denom
        metric_pooled_cos /= denom

        if args.max_grad_norm and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(student_te1.parameters(), args.max_grad_norm)

        optim.step()
        sched.step()
        global_step += 1

        if accelerator.is_main_process and (global_step % args.log_every == 0 or global_step == 1):
            dt = time.time() - t0
            lr = sched.get_last_lr()[0]
            print(f"[step {global_step:6d}/{args.train_steps}] loss={total_loss:.6f} lr={lr:.3e} elapsed={dt/60.0:.1f}m")

        if accelerator.is_main_process and (global_step % args.metrics_every == 0 or global_step == 1):
            unwrapped_student = accelerator.unwrap_model(student_te1)
            gnorm = grad_global_norm(unwrapped_student)
            print(
                "  metrics: "
                f"pooled_cos={metric_pooled_cos:.5f} "
                f"token_mse={metric_token_mse:.6g} pooled_mse={metric_pooled_mse:.6g} "
                f"token_rel_rmse={metric_token_rel:.4f} pooled_rel_rmse={metric_pooled_rel:.4f} "
                f"grad_norm={gnorm:.3f}"
            )

        if accelerator.is_main_process and (global_step % args.save_every == 0 or global_step == args.train_steps):
            save_dir = args.output_dir
            os.makedirs(save_dir, exist_ok=True)

            unwrapped = accelerator.unwrap_model(student_te1)
            unwrapped.save_pretrained(os.path.join(save_dir, "text_encoder"))
            tokenizer_1.save_pretrained(os.path.join(save_dir, "tokenizer"))

            cfg = {
                "source_teacher_dir": args.merged_model_dir,
                "images_dir": args.images_dir,
                "train_steps": args.train_steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "w_token": args.w_token,
                "w_pooled": args.w_pooled,
                "max_length": max_len,
                "mixed_precision": args.mixed_precision,
                "seed": args.seed,
                "global_step": global_step,
            }
            with open(os.path.join(save_dir, "distill_config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            print(f"[save] wrote TE1 to: {os.path.join(save_dir, 'text_encoder')}")

    if accelerator.is_main_process:
        print("[done] distillation complete.")


if __name__ == "__main__":
    main()
