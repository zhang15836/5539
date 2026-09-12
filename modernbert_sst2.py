"""
ModernBERT fine-tuning on SST-2 — head-tuning vs LoRA
Target: single NVIDIA A100-PCIE-40GB.

pip install -U transformers datasets peft accelerate matplotlib

Notes:
- GLUE SST-2's official `test` split has no public labels (all -1, reserved
  for leaderboard submission). We carve a held-out "test" set out of `train`
  (5%) and use the official `validation` split as "dev". Dev is what's used
  for model selection / early-stopping-by-best-checkpoint and for Table 1's
  "Accuracy (validation)" column; the carved test set gives a second,
  unseen-during-training number.
- Requires transformers >= 4.48 (ModernBERT support).
"""

import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
)

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LEN = 128
SEED = 42
OUTPUT_ROOT = "runs"

torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
raw = load_dataset("glue", "sst2")
split = raw["train"].train_test_split(test_size=0.05, seed=SEED)
dataset = DatasetDict(
    train=split["train"],
    validation=raw["validation"],
    test=split["test"],
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LEN)


tokenized = dataset.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2. Callback: evaluate on a fixed train subset every time we eval on dev,
#    so we can plot train vs dev accuracy on the same x-axis.
# ---------------------------------------------------------------------------
class TrainAccuracyCallback(TrainerCallback):
    def __init__(self, trainer_ref_getter, train_subset):
        self.trainer_ref_getter = trainer_ref_getter
        self.train_subset = train_subset
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Trainer.evaluate() fires on_evaluate on *every* call, including the
        # post-training dev/test report calls in run_head_tuning/run_lora --
        # those use metric_key_prefix="test" (or could be anything else), so
        # "eval_accuracy" won't be in `metrics` for them. Only react to the
        # actual epoch-boundary dev-set evals triggered during trainer.train().
        if metrics is None or "eval_accuracy" not in metrics:
            return

        trainer = self.trainer_ref_getter()
        # NOTE: use predict(), not evaluate() -- Trainer.evaluate() fires
        # on_evaluate on completion, so calling it from inside on_evaluate
        # would re-enter this callback and recurse forever. predict() only
        # fires on_predict, so it's safe here, but it also doesn't log
        # anything on its own -- we print the result ourselves below.
        train_output = trainer.predict(self.train_subset, metric_key_prefix="train")
        train_acc = train_output.metrics["train_accuracy"]
        dev_acc = metrics["eval_accuracy"]
        self.history.append({"epoch": state.epoch, "train_acc": train_acc, "dev_acc": dev_acc})
        print(f"[epoch {state.epoch:.2f}] train_accuracy={train_acc:.4f}  dev_accuracy={dev_acc:.4f}")


class PrintLogCallback(TrainerCallback):
    """Plain `print`-based periodic logging (step loss/lr + epoch summaries).

    Deliberately not tqdm-based: a carriage-return progress bar is unreadable
    once stdout is redirected to a log file on a remote box, which is the
    common case when training runs for 10+ minutes unattended on a cloud GPU.
    """

    def __init__(self, run_name):
        self.run_name = run_name
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"[{self.run_name}] training started -- {state.max_steps} steps total")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        fields = [f"step={state.global_step}/{state.max_steps}", f"epoch={state.epoch:.2f}", f"elapsed={elapsed:.0f}s"]
        for key in ("loss", "learning_rate", "eval_loss", "eval_accuracy"):
            if key in logs:
                fields.append(f"{key}={logs[key]:.4g}")
        print(f"[{self.run_name}] " + " | ".join(fields))

    def on_train_end(self, args, state, control, **kwargs):
        print(f"[{self.run_name}] training finished in {time.time() - self.start_time:.0f}s")


def build_trainer(model, run_name, train_subset_size=2000, epochs=5, lr=1e-3, bs=64, log_every=25):
    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_ROOT, run_name),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=128,
        num_train_epochs=epochs,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=log_every,  # print a log line every `log_every` steps
        disable_tqdm=True,  # plain printed logs instead, see PrintLogCallback
        bf16=True,  # A100 supports bf16 natively
        report_to="none",
        seed=SEED,
    )

    train_subset = tokenized["train"].shuffle(seed=SEED).select(range(train_subset_size))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    cb = TrainAccuracyCallback(lambda: trainer, train_subset)
    trainer.add_callback(cb)
    trainer.add_callback(PrintLogCallback(run_name))
    return trainer, cb


def plot_curves(history, title, path):
    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    dev_acc = [h["dev_acc"] for h in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_acc, marker="o", label="train")
    plt.plot(epochs, dev_acc, marker="o", label="dev (validation)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 3. 4.1 -- Head-tuning only
#    Freeze the backbone (`model.model`), train only `model.head`
#    (ModernBertPredictionHead: dense + act + norm) and `model.classifier`
#    (the newly-initialized linear layer for our 2 labels) -- these are the
#    only params added on top of the pretrained backbone.
# ---------------------------------------------------------------------------
def run_head_tuning():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    n_trainable = count_trainable_params(model)
    print(f"[head-tuning] trainable params: {n_trainable:,}")

    trainer, cb = build_trainer(model, "head_tuning", epochs=5, lr=1e-3, bs=64)
    trainer.train()
    trainer.remove_callback(cb)  # done recording the curve; avoid polluting the final report evals

    dev_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    plot_curves(
        cb.history,
        "Head-tuning: train vs dev accuracy",
        os.path.join(OUTPUT_ROOT, "head_tuning_curve.png"),
    )

    return {
        "trainable_params": n_trainable,
        "dev_accuracy": dev_metrics["eval_accuracy"],
        "test_accuracy": test_metrics["test_accuracy"],
        "history": cb.history,
    }


# ---------------------------------------------------------------------------
# 4. 4.2 -- LoRA
#    Freeze the backbone, add LoRA adapters on the attention projections
#    (`attn.Wqkv`, `attn.Wo` in every layer) and keep `classifier`/`head`
#    fully trainable via `modules_to_save` (same as 4.1). We first do a
#    cheap "dry" search over a few r values (no training, just parameter
#    counting) to find the r whose trainable-param count is closest to
#    4.1's, then train once with that r.
# ---------------------------------------------------------------------------
def build_lora_model(r):
    from peft import LoraConfig, get_peft_model, TaskType

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.05,
        target_modules=["attn.Wqkv", "attn.Wo"],
        modules_to_save=["classifier", "head"],
    )
    return get_peft_model(model, lora_config)


def pick_lora_r(target_trainable_params, candidates=(1, 2, 4, 8, 16, 32, 64)):
    best_r, best_diff = None, None
    for r in candidates:
        model = build_lora_model(r)
        n = count_trainable_params(model)
        diff = abs(n - target_trainable_params)
        print(f"  dry-run r={r}: trainable={n:,} (diff={diff:,})")
        if best_diff is None or diff < best_diff:
            best_r, best_diff = r, diff
        del model
    return best_r


def run_lora(target_trainable_params):
    print("[LoRA] searching r to match head-tuning trainable-param count...")
    r = pick_lora_r(target_trainable_params)

    model = build_lora_model(r)
    n_trainable = count_trainable_params(model)
    print(f"[LoRA r={r}] trainable params: {n_trainable:,} (target ~{target_trainable_params:,})")

    trainer, cb = build_trainer(model, f"lora_r{r}", epochs=5, lr=2e-4, bs=64)
    trainer.train()
    trainer.remove_callback(cb)  # done recording the curve; avoid polluting the final report evals

    dev_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    plot_curves(
        cb.history,
        f"LoRA (r={r}): train vs dev accuracy",
        os.path.join(OUTPUT_ROOT, "lora_curve.png"),
    )

    return {
        "r": r,
        "trainable_params": n_trainable,
        "dev_accuracy": dev_metrics["eval_accuracy"],
        "test_accuracy": test_metrics["test_accuracy"],
        "history": cb.history,
    }


# ---------------------------------------------------------------------------
# 5. Run everything, print Table 1
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    head_result = run_head_tuning()
    lora_result = run_lora(head_result["trainable_params"])

    print("\n=== Table 1: Results of training ModernBERT on SST2 ===")
    print(f"{'Approach':<15}{'Trainable params':<20}{'Accuracy (validation)':<25}{'Accuracy (test)'}")
    print(
        f"{'Head tuning':<15}{head_result['trainable_params']:<20,}"
        f"{head_result['dev_accuracy']:<25.4f}{head_result['test_accuracy']:.4f}"
    )
    print(
        f"{'LoRA (r=' + str(lora_result['r']) + ')':<15}{lora_result['trainable_params']:<20,}"
        f"{lora_result['dev_accuracy']:<25.4f}{lora_result['test_accuracy']:.4f}"
    )

    with open(os.path.join(OUTPUT_ROOT, "results.json"), "w") as f:
        json.dump({"head_tuning": head_result, "lora": lora_result}, f, indent=2)

