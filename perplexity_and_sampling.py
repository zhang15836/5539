"""
Assignment 2.3 — Measuring Perplexity and Sampling Strategies
Model: DistilGPT2 (HuggingFace)

Runs on CPU or GPU automatically (uses CUDA / A100 if available).

Usage:
    python main.py
"""

from __future__ import annotations

import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_NAME = "distilgpt2"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

set_seed(SEED)
random.seed(SEED)

print(f"Loading {MODEL_NAME} on device: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ---------------------------------------------------------------------------
# (a) Perplexity Analysis
# ---------------------------------------------------------------------------


def compute_perplexity(text: str) -> float:
    """Perplexity of `text` under the model, via mean token NLL (loss)."""
    encodings = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    # outputs.loss is the mean per-token cross-entropy (natural log), already
    # averaged over the (shifted) sequence -> ppl = exp(loss)
    return torch.exp(outputs.loss).item()


paragraph = (
    "The old lighthouse stood alone on the rocky cliff, its light sweeping "
    "across the dark water every ten seconds. Sailors who passed that coast "
    "said the keeper never slept, and that the beam had never once failed "
    "in a hundred years. On stormy nights, fishermen claimed they could see "
    "a second light flickering just behind the first, as if someone else "
    "was watching over the sea."
)

# Shuffle at the WORD level so it stays valid, tokenizable English text
# but destroys syntax/semantics -> should raise perplexity sharply.
words = paragraph.split()
shuffled_words = words.copy()
random.shuffle(shuffled_words)
shuffled_paragraph = " ".join(shuffled_words)

ppl_original = compute_perplexity(paragraph)
ppl_shuffled = compute_perplexity(shuffled_paragraph)

print("\n" + "=" * 80)
print("(a) PERPLEXITY ANALYSIS")
print("=" * 80)
print(f"\nOriginal paragraph:\n{paragraph}")
print(f"\nPerplexity (original): {ppl_original:.2f}")
print(f"\nShuffled paragraph:\n{shuffled_paragraph}")
print(f"\nPerplexity (shuffled): {ppl_shuffled:.2f}")
print(f"\nRatio (shuffled / original): {ppl_shuffled / ppl_original:.2f}x")

# ---------------------------------------------------------------------------
# (b) Sampling Comparison
# ---------------------------------------------------------------------------

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 150
TEMPERATURES = [0, 0.3, 0.6, 0.9, 1.2, 1.5]


def distinct_n(text: str, n: int) -> float:
    """Distinct-n: ratio of unique n-grams to total n-grams (diversity proxy)."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def generate(prompt: str, do_sample: bool, temperature: float | None = None):
    set_seed(SEED)  # keep decoding comparable across strategies
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = 0  # pure temperature sampling, no top-k truncation
        gen_kwargs["top_p"] = 1.0
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


print("\n" + "=" * 80)
print("(b) SAMPLING COMPARISON")
print("=" * 80)

results = {}

print(f"\n--- Greedy decoding ---")
greedy_text = generate(PROMPT, do_sample=False)
print(greedy_text)
results["greedy"] = {
    "text": greedy_text,
    "distinct_1": distinct_n(greedy_text, 1),
    "distinct_2": distinct_n(greedy_text, 2),
}

for T in TEMPERATURES:
    print(f"\n--- Temperature sampling, T={T} ---")
    if T == 0:
        # T=0 is mathematically degenerate for softmax (division by zero);
        # in practice it collapses to argmax, i.e. identical to greedy decoding.
        text = generate(PROMPT, do_sample=False)
        note = "(T=0 is equivalent to greedy decoding — softmax with T->0 collapses to argmax)"
    else:
        text = generate(PROMPT, do_sample=True, temperature=T)
        note = ""
    print(text, note)
    results[f"T={T}"] = {
        "text": text,
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
    }

print("\n" + "=" * 80)
print("DIVERSITY SUMMARY (distinct-1 / distinct-2 — higher = more lexical variety)")
print("=" * 80)
for key, r in results.items():
    print(f"{key:12s} distinct-1={r['distinct_1']:.3f}  distinct-2={r['distinct_2']:.3f}")

# ---------------------------------------------------------------------------
# Save everything to disk
# ---------------------------------------------------------------------------

summary = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "perplexity": {
        "original_paragraph": paragraph,
        "shuffled_paragraph": shuffled_paragraph,
        "ppl_original": ppl_original,
        "ppl_shuffled": ppl_shuffled,
        "ratio": ppl_shuffled / ppl_original,
    },
    "sampling": results,
}

with open("results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved full results to results.json")

