from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import wandb
from unsloth import FastLanguageModel
from torch.utils.data import Dataset, DataLoader
from collections import Counter


os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

FIT_DIR = Path(__file__).resolve().parent
CENTAUR_DIR = FIT_DIR.parent
REPO_ROOT = CENTAUR_DIR.parent
for d in (str(CENTAUR_DIR), str(REPO_ROOT)):
    sys.path.insert(0, d) if d not in sys.path else None

from config.constants import CHOICES, MAIN_TEXT, BERNOULLI_TEXT, LINCON_TEXT, QUESTION

DATA_DIR = FIT_DIR / "data"
# --- Use the fully merged Centaur model (Option B) ---
MODEL_NAME = "marcelbinz/Llama-3.1-Centaur-70B"
MAX_SEQ = 1024
BATCH = 4
GRAD_ACCUM = 2
LR = 2e-5
EPOCHS = 2
EVAL_STEPS = 500
EVAL_N = 500
WARMUP_RATIO = 0.05
LABEL_SMOOTH = 0.05  # reduced from 0.2

# --- LoRA config ---
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

CONTEXT_MAP = {0: "square", 1: "circle", 2: "triangle"}
CONTEXT_INSTRUCTION = {
    "two_context": " You will also see either a square or a circle.",
    "three_context": " You will also see either a square, a circle, or a triangle.",
}
NARMS = 5
NFEATS = 3


# --- Prompt construction (mirrors mab.py exactly) ---


def _base_text(gtype):
    if gtype == "bernoulli":
        return BERNOULLI_TEXT
    if gtype == "linear_contextual":
        return LINCON_TEXT
    if gtype in CONTEXT_INSTRUCTION:
        return MAIN_TEXT + CONTEXT_INSTRUCTION[gtype]
    return MAIN_TEXT


def _format_lincon_features(all_arm_fts):
    return "".join(
        f"Button {i + 1} features: {np.array(ft, dtype=int)}\n"
        for i, ft in enumerate(all_arm_fts)
    )


def build_pairs(game):
    gtype = game["type"]
    instruction = _base_text(gtype) + "\n\nHint: " + game["hint"] + "\n"
    contexts = game.get("contexts")
    all_arm_fts = game.get("allArmFts")
    arm_fts = game.get("armFts")
    pairs, history = [], ""
    for t, (ch, rw) in enumerate(zip(game["choices"], game["rewards"])):
        c1 = min(5, max(1, ch + 1))
        if gtype == "linear_contextual" and all_arm_fts:
            str_feats = _format_lincon_features(all_arm_fts[t])
            chosen_ft = all_arm_fts[t][ch]
            pairs.append((instruction + history + str_feats + QUESTION, str(c1)))
            history += f"Trial {t}: features: {np.array(chosen_ft, dtype=int)}, choice: <<{c1}>>, points: {rw:.2f}.\n"
        elif contexts is not None:
            shape = CONTEXT_MAP.get(contexts[t], "square")
            pairs.append(
                (instruction + f"You see a {shape}.\n" + history + QUESTION, str(c1))
            )
            history += (
                f"Trial {t}: shape: {shape}, choice: <<{c1}>>, points: {rw:.2f}.\n"
            )
        else:
            pairs.append((instruction + history + QUESTION, str(c1)))
            history += f"Trial {t}: choice: <<{c1}>>, points: {rw:.2f}.\n"
    return pairs


# --- Dataset & collation ---


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=MAX_SEQ):
        self.pairs, self.tok, self.ml = pairs, tokenizer, max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        prompt, comp = self.pairs[i]
        p_ids = self.tok(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.ml - 20,
            return_tensors=None,
        )["input_ids"]
        c_ids = (
            self.tok(comp, add_special_tokens=False, return_tensors=None)["input_ids"]
            or self.tok(" ", add_special_tokens=False, return_tensors=None)["input_ids"]
        )
        ids = (p_ids + c_ids)[: self.ml]
        labels = ([-100] * len(p_ids) + c_ids)[: self.ml]
        return {"input_ids": ids, "labels": labels, "length": len(ids)}


def collate(batch, pad_id, max_len=MAX_SEQ):
    ml = min(max_len, max(b["length"] for b in batch))

    def pad(seq, fill):
        return seq + [fill] * (ml - len(seq))

    return {
        "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
        "attention_mask": torch.tensor([pad([1] * b["length"], 0) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
    }


# --- Model loading ---


def load_model(name=MODEL_NAME, max_seq=MAX_SEQ, lora_r=LORA_R, lora_alpha=LORA_ALPHA):
    # Load the fully merged Centaur model
    model, tok = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    # Attach a FRESH LoRA adapter for our fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=LORA_TARGETS,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # let unsloth handle GC efficiently
        random_state=42,
    )

    # Print trainable param count for sanity check
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )

    pad_id = tok.pad_token_id or tok.eos_token_id or 0
    tok.pad_token_id = pad_id
    return model, tok, pad_id


def _choice_ids(tok):
    ids = []
    for c in CHOICES:
        t = tok.encode(c, add_special_tokens=False)
        ids.append(
            t[0] if len(t) == 1 else tok.encode(" " + c, add_special_tokens=False)[0]
        )
    return ids


# --- Eval ---


def run_eval(model, tok, pairs, device, max_seq=MAX_SEQ, temp=0.6):
    model.eval()
    cids = _choice_ids(tok)
    correct = ll = n = 0
    with torch.no_grad():
        for prompt, comp in tqdm(pairs, desc="Eval"):
            inp = tok(
                prompt, return_tensors="pt", truncation=True, max_length=max_seq
            ).to(device)
            logits = model(**inp).logits[:, -1, cids].float()
            probs = torch.softmax(logits / temp, dim=-1).cpu().numpy().ravel()
            idx = CHOICES.index(comp) if comp in CHOICES else 0
            ll += np.log(np.clip(probs[idx], 1e-12, 1.0))
            correct += int(np.argmax(probs) == idx)
            n += 1
    return (correct / n if n else 0.0), (ll / n if n else 0.0)


# --- Loss ---


def smooth_choice_loss(logits, labels, cids, smooth=LABEL_SMOOTH):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if not mask.any():
        return logits.new_tensor(0.0)
    active = shift_logits[mask][:, cids].float()  # (N, 5)
    tgt_ids = shift_labels[mask]
    id2i = {c: i for i, c in enumerate(cids)}
    tgt = torch.tensor([id2i.get(t.item(), 0) for t in tgt_ids], device=logits.device)
    soft = torch.full_like(active, smooth / (len(cids) - 1))
    soft.scatter_(1, tgt.unsqueeze(1), 1.0 - smooth)
    return -(soft * torch.log_softmax(active, dim=1)).sum(1).mean()


# --- Training ---


def train(
    model,
    tok,
    loader,
    eval_pairs,
    device,
    smooth=LABEL_SMOOTH,
    epochs=EPOCHS,
    grad_accum=GRAD_ACCUM,
    lr=LR,
    max_seq=MAX_SEQ,
    eval_steps=EVAL_STEPS,
    warmup_ratio=WARMUP_RATIO,
):
    cids = _choice_ids(tok)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )

    # Compute total training steps for scheduler
    total_steps = (len(loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    # Cosine LR schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    acc0, ll0 = run_eval(model, tok, eval_pairs, device, max_seq)
    print(f"\nPre-train eval: acc={acc0:.4f}, ll={ll0:.4f}")
    print(f"Label smoothing: {smooth}")
    wandb.log({"eval/acc": acc0, "eval/ll": ll0, "step": 0})
    torch.cuda.empty_cache()

    model.train()
    step = 0
    best_ll = float("-inf")

    for epoch in range(epochs):
        total, nb = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch = {
                k: v.to(device).clone() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels = batch.pop("labels")
            out = model(**batch, use_cache=False)
            loss = smooth_choice_loss(out.logits, labels, cids, smooth)
            (loss / grad_accum).backward()
            total += loss.item()
            nb += 1
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{total / nb:.4f}",
                lr=f"{current_lr:.2e}",
            )
            if nb % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad()
                step += 1
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/avg_loss": total / nb,
                        "train/lr": current_lr,
                        "epoch": epoch,
                        "step": step,
                    }
                )
                if step % eval_steps == 0 and eval_pairs:
                    a, l = run_eval(model, tok, eval_pairs, device, max_seq)
                    print(f"  step {step}  acc={a:.4f}  ll={l:.4f}")
                    wandb.log({"eval/acc": a, "eval/ll": l, "step": step})
                    torch.cuda.empty_cache()
                    model.train()

                    if l > best_ll:
                        best_ll = l
                        model.save_pretrained(str(FIT_DIR / "best_checkpoint"))
                        print(f"  *** New best ll={l:.4f}, saved checkpoint ***")
        print(f"  Epoch {epoch + 1}  avg_loss={total / max(1, nb):.4f}")
        wandb.log({"epoch_loss": total / max(1, nb), "epoch": epoch + 1})

    acc1, ll1 = run_eval(model, tok, eval_pairs, device, max_seq)
    print(f"\nPost-train eval: acc={acc1:.4f}, ll={ll1:.4f}")
    wandb.log({"eval/acc": acc1, "eval/ll": ll1, "step": step})
    return acc1, ll1


# --- Main ---


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument("--out-dir", type=Path, default=Path("./lora_out"))
    p.add_argument("--push-to-hub", type=str, default="eys8549/centaur_IBT")
    p.add_argument("--max-seq", type=int, default=MAX_SEQ)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--eval-steps", type=int, default=EVAL_STEPS)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--lora-r", type=int, default=LORA_R)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--wandb-project", type=str, default="centaur-lora")
    p.add_argument("--wandb-name", type=str, default=None)
    args = p.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "model": args.model_name,
            "lr": args.lr,
            "epochs": EPOCHS,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "label_smoothing": args.label_smoothing,
            "max_seq": args.max_seq,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "warmup_ratio": args.warmup_ratio,
        },
    )

    model, tok, pad_id = load_model(
        args.model_name,
        args.max_seq,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Build pairs, skipping types we can't reconstruct prompts for
    train_games = [
        g for g in json.load(open(args.data_dir / "train_games.json")) if g.get("type")
    ]
    test_games = [
        g for g in json.load(open(args.data_dir / "test_games.json")) if g.get("type")
    ]
    train_pairs = [p for g in train_games for p in build_pairs(g)]
    eval_pairs = [p for g in test_games for p in build_pairs(g)]
    if len(eval_pairs) > EVAL_N:
        random.seed(42)
        eval_pairs = random.sample(eval_pairs, EVAL_N)

    loader = DataLoader(
        PairDataset(train_pairs, tok, args.max_seq),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate(b, pad_id, args.max_seq),
    )

    device = next(model.parameters()).device
    train(
        model,
        tok,
        loader,
        eval_pairs,
        device,
        smooth=args.label_smoothing,
        grad_accum=args.grad_accum,
        max_seq=args.max_seq,
        eval_steps=args.eval_steps,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
    )

    print(f"Pushing to {args.push_to_hub}...")
    try:
        token = os.environ.get("HF_TOKEN")
        model.push_to_hub(args.push_to_hub, safe_serialization=True, token=token)
        tok.push_to_hub(args.push_to_hub, token=token)
    except Exception as e:
        print(f"Hub push failed ({e}), saving locally...")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(args.out_dir))
        tok.save_pretrained(str(args.out_dir))

    wandb.finish()


if __name__ == "__main__":
    main()
