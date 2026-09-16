"""
Training script for Learnable Trust-Boundary Delimiters (LTBD).

Steps:
  1. Load self-labeled data.
  2. Construct D' with approximately 50% clean and 50% injected samples.
  3. Register four delimiter special tokens and resize token embeddings.
  4. Freeze all model parameters and optimize only the four delimiter embeddings.
  5. Train with CE loss and preference loss on assistant responses.
  6. Export the learned delimiter embeddings to JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from injections import InjectionConfig, apply_injection, sample_variant
from prompting import (
    DELIMITER_TOKENS,
    build_plain_messages,
    register_delimiter_tokens,
    tokenize_chat_for_sft,
)

def _model_chat_template_path(model_dir: str) -> str:
    return os.path.join(model_dir, "chat_template.jinja")

def _repo_chat_template_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_template.jinja")

def _load_chat_template(tokenizer: Any, *, model_dir: str) -> None:
    """
    Prefer loading template from the model directory (if present), otherwise fall
    back to the repo template. This keeps training/eval prompt formatting aligned
    when resuming from a previously-saved checkpoint directory.
    """
    model_tmpl = _model_chat_template_path(model_dir)
    if os.path.isfile(model_tmpl):
        with open(model_tmpl, "r") as f:
            tokenizer.chat_template = f.read()
        print(f"[Step 1] chat_template loaded from model: {model_tmpl}")
        return
    repo_tmpl = _repo_chat_template_path()
    if os.path.isfile(repo_tmpl):
        with open(repo_tmpl, "r") as f:
            tokenizer.chat_template = f.read()
        print(f"[Step 1] chat_template loaded from repo: {repo_tmpl}")

def load_ce_jsonl(path: str, *, max_samples: Optional[int] = None) -> List["TrainRow"]:
    """
    Load CE defensive dataset from JSONL.
    Expected fields per row:
      - instruction
      - data
      - self_response (preferred) OR chosen (fallback)
      - optional: pref_weight
      - optional: variant
    """
    rows: List[TrainRow] = []
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            instruction = obj.get("instruction")
            data = obj.get("data")
            target = obj.get("self_response")
            if target is None:
                target = obj.get("chosen")
            if instruction is None or data is None or target is None:
                continue
            rows.append(
                TrainRow(
                    instruction=str(instruction),
                    data=str(data),
                    target=str(target),
                    rejected=str(target),  # CE-only rows => pref term becomes ~0
                    pref_weight=float(obj.get("pref_weight", 0.0)),
                )
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No usable CE rows read from {path}")
    return rows

def _copy_delimiter_embeddings_from_model(
    *,
    dst_model: Any,
    dst_tokenizer: Any,
    delimiter_ids: List[int],
    src_model_dir: str,
) -> None:
    """
    Copy delimiter token embedding vectors from a source HF model directory into dst_model.
    Tokens are matched by token string in DELIMITER_TOKENS (not by id).
    """
    src_tok = AutoTokenizer.from_pretrained(src_model_dir, use_fast=True)
    src_model = AutoModelForCausalLM.from_pretrained(
        src_model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    src_emb = src_model.get_input_embeddings().weight.detach()
    dst_emb = dst_model.get_input_embeddings().weight
    with torch.no_grad():
        for tok_str, dst_id in zip(DELIMITER_TOKENS, delimiter_ids):
            src_id = src_tok.convert_tokens_to_ids(tok_str)
            if src_id is None or int(src_id) < 0:
                raise ValueError(f"Source model missing delimiter token {tok_str!r}: {src_model_dir}")
            dst_emb[int(dst_id)].copy_(src_emb[int(src_id)].to(dst_emb.dtype).to(dst_emb.device))
    print(f"[Step 3] Initialized delimiter embeddings from: {src_model_dir}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainRow:
    instruction: str
    data: str
    target: str
    rejected: str
    pref_weight: float


class DefensiveDataset(Dataset):
    def __init__(self, rows: List[TrainRow], tokenizer: Any, *, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        chosen = tokenize_chat_for_sft(
            self.tokenizer, r.instruction, r.data, r.target, max_length=self.max_length
        )
        rejected = tokenize_chat_for_sft(
            self.tokenizer, r.instruction, r.data, r.rejected, max_length=self.max_length
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
            "pref_weight": float(r.pref_weight),
        }


class PadCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["chosen_input_ids"]) for f in features)
        max_len_rej = max(len(f["rejected_input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        c_input_ids, c_attn, c_labels = [], [], []
        r_input_ids, r_attn, r_labels = [], [], []
        pref_w = []
        for f in features:
            pad = max_len - len(f["chosen_input_ids"])
            c_input_ids.append(f["chosen_input_ids"] + [pad_id] * pad)
            c_attn.append(f["chosen_attention_mask"] + [0] * pad)
            c_labels.append(f["chosen_labels"] + [-100] * pad)

            pad_r = max_len_rej - len(f["rejected_input_ids"])
            r_input_ids.append(f["rejected_input_ids"] + [pad_id] * pad_r)
            r_attn.append(f["rejected_attention_mask"] + [0] * pad_r)
            r_labels.append(f["rejected_labels"] + [-100] * pad_r)
            pref_w.append(float(f.get("pref_weight", 1.0)))

        return {
            "chosen_input_ids": torch.tensor(c_input_ids, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(c_attn, dtype=torch.long),
            "chosen_labels": torch.tensor(c_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(r_input_ids, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(r_attn, dtype=torch.long),
            "rejected_labels": torch.tensor(r_labels, dtype=torch.long),
            "pref_weight": torch.tensor(pref_w, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Load self-labeled data
# ---------------------------------------------------------------------------
def load_self_labeled(path: str, *, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            if obj.get("instruction") is None or obj.get("data") is None or obj.get("self_response") is None:
                continue
            rows.append({
                "instruction": str(obj["instruction"]),
                "data": str(obj["data"]),
                "self_response": str(obj["self_response"]),
            })
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No usable rows read from {path}")
    return rows


def load_pref_jsonl(path: str, *, max_samples: Optional[int] = None) -> List[TrainRow]:
    rows: List[TrainRow] = []
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            if obj.get("instruction") is None or obj.get("data") is None:
                continue
            if obj.get("chosen") is None or obj.get("rejected") is None:
                continue
            rows.append(
                TrainRow(
                    instruction=str(obj["instruction"]),
                    data=str(obj["data"]),
                    target=str(obj["chosen"]),
                    rejected=str(obj["rejected"]),
                    pref_weight=float(obj.get("pref_weight", 1.0)),
                )
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No usable preference rows read from {path}")
    return rows


def export_dprime_chosen_jsonl(path: str, rows: List[TrainRow]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        for r in rows:
            writer.write(
                {
                    "instruction": r.instruction,
                    "data": r.data,
                    "chosen": r.target,
                    "variant": "unknown",
                    "pref_weight": float(r.pref_weight),
                }
            )


# ---------------------------------------------------------------------------
# Build D': 50% clean, 50% injected (weighted sampling over 3 variants)
# ---------------------------------------------------------------------------
def build_D_prime(
    self_labeled: List[Dict[str, str]],
    *,
    seed: int,
    injection_cfg: InjectionConfig,
) -> List[TrainRow]:
    rng = random.Random(seed)
    n = len(self_labeled)
    idxs = list(range(n))
    rng.shuffle(idxs)
    half = n // 2
    clean_idxs = set(idxs[:half])

    rows: List[TrainRow] = []
    n_clean, n_injected = 0, 0
    for i, ex in enumerate(self_labeled):
        instruction = ex["instruction"]
        data_clean = ex["data"]
        target = ex["self_response"]
        # Placeholder; filled later by base-plain generation.
        rejected = ""

        if i in clean_idxs:
            rows.append(
                TrainRow(
                    instruction=instruction,
                    data=data_clean,
                    target=target,
                    rejected=target,  # for clean rows we can reuse chosen as rejected
                    pref_weight=0.0,
                )
            )
            n_clean += 1
        else:
            if not data_clean:
                rows.append(
                    TrainRow(
                        instruction=instruction,
                        data=data_clean,
                        target=target,
                        rejected=target,
                        pref_weight=0.0,
                    )
                )
                n_clean += 1
                continue

            j = rng.randrange(n)
            while j == i and n > 1:
                j = rng.randrange(n)
            other_ex = self_labeled[j]

            variant = sample_variant(rng, injection_cfg)
            data_attacked = apply_injection(
                data_clean, variant=variant, other_example=other_ex, rng=rng, cfg=injection_cfg,
            )
            rows.append(
                TrainRow(
                    instruction=instruction,
                    data=data_attacked,
                    target=target,
                    rejected=rejected,
                    pref_weight=1.0,
                )
            )
            n_injected += 1

    print(f"  D' stats: {n_clean} clean + {n_injected} injected = {len(rows)} total")
    return rows


# ---------------------------------------------------------------------------
# Export: save the 4 delimiter token embeddings to JSON
# ---------------------------------------------------------------------------
def export_delimiter_embeddings(
    model: Any,
    tokenizer: Any,
    *,
    model_id: str,
    output_path: str,
) -> None:
    embed_weight = model.get_input_embeddings().weight.detach().float().cpu()
    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in DELIMITER_TOKENS]

    entry: Dict[str, list] = {}
    for tok_str, tok_id in zip(DELIMITER_TOKENS, token_ids):
        entry[tok_str] = embed_weight[tok_id].tolist()

    payload: Dict[str, Any] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                payload = json.load(f) or {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

    payload[model_id] = entry
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"Exported delimiter embeddings → {output_path}")


# ---------------------------------------------------------------------------
# Custom Trainer: gradient mask to only update delimiter embeddings
# ---------------------------------------------------------------------------
class DelimiterOnlyTrainer(Trainer):
    """Zero out gradients for all non-delimiter embedding rows after each backward."""

    def __init__(self, *args, delimiter_ids: List[int], input_embeddings: Any, **kwargs):
        super().__init__(*args, **kwargs)
        self._delimiter_ids = set(delimiter_ids)
        self._input_embeddings = input_embeddings

    def training_step(self, *a, **kw):
        loss = super().training_step(*a, **kw)
        emb_grad = self._input_embeddings.weight.grad
        if emb_grad is not None:
            vocab_size = emb_grad.shape[0]
            mask = torch.ones(vocab_size, 1, device=emb_grad.device, dtype=emb_grad.dtype)
            for tid in self._delimiter_ids:
                mask[tid] = 0.0
            emb_grad.mul_(1.0 - mask)
        return loss


class DelimiterPreferenceTrainer(DelimiterOnlyTrainer):
    def __init__(
        self,
        *args,
        pref_lambda: float,
        pref_beta: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._pref_lambda = float(pref_lambda)
        self._pref_beta = float(pref_beta)

    @staticmethod
    def _sequence_logp_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, T, V], labels: [B, T] with -100 masked.
        # Shift to align next-token prediction.
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        labels_clamped = labels.clamp(min=0)
        token_logp = log_probs.gather(dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * (labels != -100).to(token_logp.dtype)
        return token_logp.sum(dim=-1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        c_ids = inputs["chosen_input_ids"]
        c_attn = inputs["chosen_attention_mask"]
        c_labels = inputs["chosen_labels"]
        r_ids = inputs["rejected_input_ids"]
        r_attn = inputs["rejected_attention_mask"]
        r_labels = inputs["rejected_labels"]
        pref_weight = inputs.get("pref_weight")

        c_out = model(input_ids=c_ids, attention_mask=c_attn)
        r_out = model(input_ids=r_ids, attention_mask=r_attn)

        # CE on chosen assistant tokens (same masking behavior as SFT).
        ce_loss = F.cross_entropy(
            c_out.logits[:, :-1, :].reshape(-1, c_out.logits.size(-1)),
            c_labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        c_logp = self._sequence_logp_from_logits(c_out.logits, c_labels)
        r_logp = self._sequence_logp_from_logits(r_out.logits, r_labels)
        delta = c_logp - r_logp
        per_sample_pref = -F.logsigmoid(self._pref_beta * delta)
        if pref_weight is None:
            pref_loss = per_sample_pref.mean()
        else:
            w = pref_weight.to(per_sample_pref.device, dtype=per_sample_pref.dtype)
            denom = w.sum().clamp_min(1.0)
            pref_loss = (per_sample_pref * w).sum() / denom

        loss = ce_loss + self._pref_lambda * pref_loss
        if return_outputs:
            return loss, {"chosen": c_out, "rejected": r_out}
        return loss


def _generate_rejected_base_plain(
    model: Any,
    tokenizer: Any,
    rows: List[TrainRow],
    *,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    model.eval()
    device = getattr(model, "device", None)
    out_texts: List[str] = []
    with torch.no_grad():
        for r in rows:
            msgs = build_plain_messages(r.instruction, r.data)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if device is not None:
                enc = {k: v.to(device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Strip prompt part.
            gen_ids = gen[0][enc["input_ids"].shape[1] :]
            out_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return out_texts


def _latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return latest checkpoint path under output_dir, or None if missing."""
    if not os.path.isdir(output_dir):
        return None
    cands = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        suffix = name.replace("checkpoint-", "", 1)
        if suffix.isdigit():
            cands.append((int(suffix), os.path.join(output_dir, name)))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    return cands[-1][1]


def _save_loss_curves(trainer: Trainer, output_dir: str) -> None:
    """Save loss curve to CSV + PNG from trainer.state.log_history."""
    rows = []
    for entry in getattr(trainer.state, "log_history", []) or []:
        if "loss" not in entry:
            continue
        step = entry.get("step")
        epoch = entry.get("epoch")
        loss = entry.get("loss")
        if step is None or loss is None:
            continue
        rows.append((int(step), float(epoch) if epoch is not None else None, float(loss)))

    if not rows:
        return

    rows.sort(key=lambda x: x[0])
    csv_path = os.path.join(output_dir, "loss_curve.csv")
    with open(csv_path, "w") as f:
        f.write("step,epoch,loss\n")
        for step, epoch, loss in rows:
            f.write(f"{step},{'' if epoch is None else epoch},{loss}\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r[0] for r in rows]
        losses = [r[2] for r in rows]
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, losses, linewidth=1.5)
        plt.title("Training loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        png_path = os.path.join(output_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
    except Exception as e:  # noqa: BLE001
        print(f"[Warn] Failed to save loss_curve.png (matplotlib). CSV saved. Error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local snapshot path")
    ap.add_argument("--self_labeled", required=True, help="JSONL from self_label.py")
    ap.add_argument("--pref_data", default=None, help="Optional preference JSONL from gen_rejected.py")
    ap.add_argument("--ce_data", default=None, help="Optional CE JSONL for phase3 (instruction/data/self_response).")
    ap.add_argument("--export_dprime", default=None, help="If set, export fixed D' (instruction/data/chosen) JSONL and exit.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--export_json", default=None,
                    help="Path to merge/update delimiter embeddings (default: <output_dir>/delimiter_embeddings.json)")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--pref_lambda", type=float, default=1.0, help="Weight for preference loss term.")
    ap.add_argument("--pref_beta", type=float, default=0.1, help="Inverse temperature for preference loss.")
    ap.add_argument("--rejected_max_new_tokens", type=int, default=1024)
    ap.add_argument("--rejected_temperature", type=float, default=0.8)
    ap.add_argument(
        "--init_delimiters_from_model",
        default=None,
        help="If set, copy delimiter embeddings from this HF model dir before training.",
    )
    ap.add_argument(
        "--init_delimiters_random",
        action="store_true",
        help="If set, random-initialize delimiter embeddings (default: keep loaded embeddings).",
    )
    ap.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="Checkpoint path, 'auto' (default: resume latest in output_dir if exists), or 'none'.",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- Step 1: Load tokenizer, register 4 delimiter tokens ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    delimiter_ids = register_delimiter_tokens(tokenizer)
    _load_chat_template(tokenizer, model_dir=args.model)
    print(f"[Step 1] Delimiter tokens registered: {list(zip(DELIMITER_TOKENS, delimiter_ids))}")

    # ---- Step 2: Load model, resize embeddings ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"[Step 2] Model loaded, vocab resized to {len(tokenizer)}")

    # ---- Step 3: Freeze all, then unfreeze only delimiter embeddings ----
    for param in model.parameters():
        param.requires_grad = False

    input_embeddings = model.get_input_embeddings()
    with torch.no_grad():
        if args.init_delimiters_from_model:
            _copy_delimiter_embeddings_from_model(
                dst_model=model,
                dst_tokenizer=tokenizer,
                delimiter_ids=delimiter_ids,
                src_model_dir=args.init_delimiters_from_model,
            )
        elif args.init_delimiters_random:
            for tok_id in delimiter_ids:
                input_embeddings.weight[tok_id].normal_(mean=0.0, std=1.0)
            print("[Step 3] Random-initialized delimiter embeddings (init_delimiters_random=1)")
        else:
            print("[Step 3] Keeping loaded delimiter embeddings (no random re-init).")

    input_embeddings.weight.requires_grad = True

    trainable_params = len(delimiter_ids) * input_embeddings.weight.shape[1]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Step 3] Trainable: {trainable_params} / {total_params} "
          f"({trainable_params / total_params * 100:.4f}%)")

    # ---- Step 4: Build training rows ----
    if args.ce_data:
        print(f"[Step 4] Loading CE data: {args.ce_data}")
        train_rows = load_ce_jsonl(args.ce_data, max_samples=args.max_samples)
        ds = DefensiveDataset(train_rows, tokenizer, max_length=args.max_length)
        print(f"[Step 4] CE rows: {len(train_rows)}")
    elif args.pref_data:
        print(f"[Step 4] Loading preference data: {args.pref_data}")
        train_rows = load_pref_jsonl(args.pref_data, max_samples=args.max_samples)
        ds = DefensiveDataset(train_rows, tokenizer, max_length=args.max_length)
        print(f"[Step 4] Preference rows: {len(train_rows)}")
    else:
        self_labeled = load_self_labeled(args.self_labeled, max_samples=args.max_samples)
        train_rows = build_D_prime(self_labeled, seed=args.seed, injection_cfg=InjectionConfig())

        if args.export_dprime:
            print(f"[Step 4] Exporting fixed D' → {args.export_dprime}")
            export_dprime_chosen_jsonl(args.export_dprime, train_rows)
            return

        # ---- Step 4.5: Generate rejected responses using base-plain prompts ----
        print("[Step 4.5] Generating rejected (base-plain) responses …")
        rejected_texts = _generate_rejected_base_plain(
            model,
            tokenizer,
            train_rows,
            max_new_tokens=args.rejected_max_new_tokens,
            temperature=args.rejected_temperature,
        )
        train_rows = [
            TrainRow(
                instruction=r.instruction,
                data=r.data,
                target=r.target,
                rejected=rej,
                pref_weight=r.pref_weight,
            )
            for r, rej in zip(train_rows, rejected_texts)
        ]
        ds = DefensiveDataset(train_rows, tokenizer, max_length=args.max_length)
        print(f"[Step 4] D' built: {len(train_rows)} rows")

    # ---- Step 5: Train ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
    )

    trainer = DelimiterPreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=PadCollator(tokenizer),
        delimiter_ids=delimiter_ids,
        input_embeddings=input_embeddings,
        pref_lambda=args.pref_lambda,
        pref_beta=args.pref_beta,
    )

    print("[Step 5] Training started …")
    resume_ckpt: Optional[str] = None
    if isinstance(args.resume_from_checkpoint, str):
        val = args.resume_from_checkpoint.strip().lower()
        if val == "auto":
            resume_ckpt = _latest_checkpoint(args.output_dir)
        elif val == "none" or val == "":
            resume_ckpt = None
        else:
            resume_ckpt = args.resume_from_checkpoint

    if resume_ckpt:
        print(f"[Step 5] Resuming from checkpoint: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        print("[Step 5] No checkpoint resume. Starting fresh training run.")
        trainer.train()
    print("[Step 5] Training complete.")

    _save_loss_curves(trainer, args.output_dir)

    # ---- Step 6: Save model + tokenizer (chat template already set in Step 1) + embeddings ----
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save effective template for reproducibility.
    model_tmpl = _model_chat_template_path(args.model)
    repo_tmpl = _repo_chat_template_path()
    out_tmpl = os.path.join(args.output_dir, "chat_template.jinja")
    if os.path.isfile(model_tmpl):
        shutil.copy2(model_tmpl, out_tmpl)
    elif os.path.isfile(repo_tmpl):
        shutil.copy2(repo_tmpl, out_tmpl)

    export_path = args.export_json or os.path.join(args.output_dir, "delimiter_embeddings.json")
    export_delimiter_embeddings(model, tokenizer, model_id=args.model, output_path=export_path)
    print(f"[Step 6] All saved to {args.output_dir}")


if __name__ == "__main__":
    main()
