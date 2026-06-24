"""
Module 1.2b — Fine-tune the Semantic Equivalence Classifier on numeric semantics

Goal: patch the specific blind spot found in validation -- the off-the-shelf
roberta-large-mnli treats different arithmetic operations (multiply vs divide,
add vs subtract) as similar just because the phrasing is similar.

Approach: a SHORT fine-tune (few epochs, small learning rate) on a focused
set of numeric contrast pairs (data/numeric_finetune_pairs.json). This is
deliberately light-touch -- we want to teach the specific distinction without
catastrophically forgetting the general NLI ability that already got 96% on
the validation set. After training we re-run the full 50-pair validation to
confirm no regression.
"""

import os
import sys
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1.module2_semantic_equivalence import (
    SemanticEquivalenceClassifier,
    load_test_pairs,
    validate,
)

LABEL_TO_ID = {"contradiction": 0, "neutral": 1, "entailment": 2}  # matches roberta-large-mnli's head order


class NumericPairsDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        encoding = self.tokenizer(
            pair["premise"],
            pair["hypothesis"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(LABEL_TO_ID[pair["label"]], dtype=torch.long)
        return item


def load_finetune_pairs():
    repo_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "numeric_finetune_pairs.json",
    )
    with open(repo_data_path, "r") as f:
        data = json.load(f)
    return data["pairs"]


def finetune(
    model_name=None,
    epochs=3,
    learning_rate=1e-5,
    batch_size=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_name or config.NLI_MODEL_NAME

    print(f"Loading {model_name} for fine-tuning on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    pairs = load_finetune_pairs()
    print(f"Loaded {len(pairs)} numeric fine-tuning pairs "
          f"({sum(1 for p in pairs if p['label']=='entailment')} entailment, "
          f"{sum(1 for p in pairs if p['label']=='contradiction')} contradiction)")

    dataset = NumericPairsDataset(pairs, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    print(f"\nFine-tuning for {epochs} epochs (lr={learning_rate}, batch_size={batch_size}) ...")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

    model.eval()

    # Save the fine-tuned checkpoint to Drive so it survives runtime resets
    save_path = os.path.join(config.CHECKPOINT_DIR, "semantic_equivalence_finetuned")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nSaved fine-tuned model to: {save_path}")

    return model, tokenizer, save_path


def main():
    print("\nMODULE 1.2b — FINE-TUNE ON NUMERIC SEMANTICS\n")

    print("=" * 60)
    print("STEP 1 — BASELINE (before fine-tuning)")
    print("=" * 60)
    baseline_clf = SemanticEquivalenceClassifier()
    pairs = load_test_pairs()
    baseline_acc, baseline_misses = validate(baseline_clf, pairs, verbose=False)
    print(f"Baseline accuracy on 50-pair validation set: {baseline_acc:.1%}")
    print(f"Baseline misses: {[m['id'] for m in baseline_misses]}\n")

    print("=" * 60)
    print("STEP 2 — FINE-TUNING")
    print("=" * 60)
    model, tokenizer, save_path = finetune()

    print()
    print("=" * 60)
    print("STEP 3 — RE-VALIDATE (after fine-tuning)")
    print("=" * 60)
    finetuned_clf = SemanticEquivalenceClassifier(model_name=save_path)
    finetuned_acc, finetuned_misses = validate(finetuned_clf, pairs, verbose=True)

    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Before fine-tuning: {baseline_acc:.1%}  (missed: {[m['id'] for m in baseline_misses]})")
    print(f"After fine-tuning:  {finetuned_acc:.1%}  (missed: {[m['id'] for m in finetuned_misses]})")

    if finetuned_acc < baseline_acc:
        print("\nWARNING: accuracy DROPPED after fine-tuning. This means the model")
        print("forgot some general NLI ability while learning the numeric distinction")
        print("(catastrophic forgetting). Consider fewer epochs or a lower learning rate.")
    elif 47 in [m["id"] for m in finetuned_misses]:
        print("\nNote: id=47 (the original numeric blind spot) is STILL missed.")
        print("May need more epochs, more diverse numeric examples, or a higher learning rate.")
    else:
        print("\nFixed the numeric blind spot without regressing on anything else. Good to proceed.")

    # Save comparison to Drive
    out_path = os.path.join(config.CHECKPOINT_DIR, "module1_2b_finetune_comparison.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "baseline_accuracy": baseline_acc,
                "baseline_missed_ids": [m["id"] for m in baseline_misses],
                "finetuned_accuracy": finetuned_acc,
                "finetuned_missed_ids": [m["id"] for m in finetuned_misses],
                "finetuned_model_path": save_path,
            },
            f,
            indent=2,
        )
    print(f"\nSaved comparison to: {out_path}")


if __name__ == "__main__":
    main()
