"""Fine-tune the NLI verifier on scientific / health fact-checking data (step 8.5).

Runs on a GPU (Colab A100: ~20-30 min for large, ~8 min for base). Starts from the
DeBERTa-v3 NLI checkpoints the engine already uses (transfer learning), trains on
data/finetune/train.jsonl, selects the best epoch on data/finetune/val.jsonl
(held-out SciFact *train* claims + HealthVer dev), and saves a model the engine
loads with TE_NLI_MODEL=ft-large / ft-base.

    python -m truth_engine.bench.training_data           # build the data first
    python scripts/finetune_verifier.py --size large      # or --size base
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,  # noqa: E402
                          Trainer, TrainingArguments)

from truth_engine.bench.training_data import OUT as DATA  # noqa: E402
from truth_engine.pipeline import nli  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=["large", "base"], default="large")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=None)          # default: 6e-6 large, 2e-5 base
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--smoke", action="store_true", help="tiny CPU check: no checkpoints, no best-epoch reload")
args = ap.parse_args()
lr = args.lr or (6e-6 if args.size == "large" else 2e-5)
random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)

src = nli.model_dir(args.size)
if not nli.is_downloaded(args.size):
    nli.download(args.size)
out = nli.model_dir(f"ft-{args.size}")
tok = AutoTokenizer.from_pretrained(src)
# The checkpoints are stored in fp16; train with fp32 master weights (bf16 autocast on GPU),
# or the first optimizer step overflows to NaN.
model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32)
label2id = {k.lower(): v for k, v in model.config.label2id.items()}     # keep the checkpoint's label order


class Pairs(torch.utils.data.Dataset):
    def __init__(self, path):
        self.rows = [json.loads(line) for line in open(path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = tok(r["premise"], r["hypothesis"], truncation=True, max_length=args.max_len)
        enc["labels"] = label2id[r["label"]]
        return enc


train, val = Pairs(DATA / "train.jsonl"), Pairs(DATA / "val.jsonl")
if args.smoke:                       # just prove the pipeline runs and stays finite
    train.rows, val.rows = train.rows[:96], val.rows[:64]
LABELS = ["entailment", "neutral", "contradiction"]


def metrics(p):
    pred, gold = p.predictions.argmax(-1), p.label_ids
    f1 = []
    for lab in LABELS:
        k = label2id[lab]
        tp = int(((pred == k) & (gold == k)).sum())
        pr, rc = tp / max(1, int((pred == k).sum())), tp / max(1, int((gold == k).sum()))
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"accuracy": float((pred == gold).mean()), "macro_f1": float(np.mean(f1))}


targs = TrainingArguments(
    output_dir=str(DATA / f"run-{args.size}"), num_train_epochs=args.epochs, learning_rate=lr,
    per_device_train_batch_size=args.batch, per_device_eval_batch_size=64, weight_decay=0.01,
    warmup_steps=max(1, int(0.1 * len(train) / args.batch * args.epochs)),     # 10% warmup (works across versions)
    eval_strategy="epoch", save_strategy="no" if args.smoke else "epoch", save_total_limit=1,
    load_best_model_at_end=not args.smoke,
    metric_for_best_model="macro_f1", bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    logging_steps=50, report_to=[], seed=args.seed)
trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
before = trainer.evaluate()
t0 = time.time()
trainer.train()
after = trainer.evaluate()
trainer.save_model(str(out))
tok.save_pretrained(str(out))
card = {"base": nli.MODELS[args.size], "size": args.size, "lr": lr, "epochs": args.epochs, "batch": args.batch,
        "max_len": args.max_len, "seed": args.seed, "train_pairs": len(train), "val_pairs": len(val),
        "val_before": {k: before[k] for k in ("eval_accuracy", "eval_macro_f1")},
        "val_after": {k: after[k] for k in ("eval_accuracy", "eval_macro_f1")},
        "train_minutes": round((time.time() - t0) / 60, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "data_stats": json.loads((DATA / "stats.json").read_text()) | {"held_out_scifact_claims": "see stats.json"}}
(out / "training_card.json").write_text(json.dumps(card, indent=1))
print(json.dumps(card["val_before"]), "->", json.dumps(card["val_after"]), f"saved to {out}")
