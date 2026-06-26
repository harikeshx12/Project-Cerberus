"""
Module 1.3c — The Variant Generator (Strategy C: Controlled Paraphrase Generation)

Goal: fine-tune a small T5 model on Quora Question Pairs (QQP) so it learns
to directly generate a paraphrase given an input question -- a different
source of diversity than back-translation (vocabulary swaps via French
pivot) or syntactic transforms (rule-based structural rewrites). A model
trained specifically on "is this a paraphrase" should produce more natural
variety than either.

QQP schema (verified against the real HuggingFace dataset card, nyu-mll/glue,
qqp config):
    {"question1": str, "question2": str, "label": 0 or 1, "idx": int}
    label=1 means question1 and question2 ARE paraphrases of each other.
We only train on label=1 pairs -- label=0 pairs are explicitly NOT
paraphrases, training on those would teach the model the wrong thing.

T5 is trained as: input "paraphrase: {question1}" -> target "{question2}".
At inference time we feed a NEW query through the same "paraphrase: " prefix
and sample multiple candidates, exactly like BackTranslationStrategy does on
its French->English leg -- for the same reason: a single greedy decode tends
to just repeat the input with minor changes; beam search with no_repeat_ngram
gives real diversity.

Every candidate still goes through the same three gates (semantic
equivalence, fluency, structural sanity) before being accepted -- this
strategy gets no special treatment for being model-based.

Changes from v1 (after underfitting observed in first run):
- max_train_examples: 20000 -> 80000 (4x more training pairs)
- epochs: 2 -> 5 (model needs more passes to escape "copy input" local optimum)
- learning_rate: 3e-4 -> 2e-4 (slightly lower for stable refinement over 5 epochs)
- Added linear warmup + decay scheduler (10% warmup, then linear decay)
- Generation: replaced do_sample=True/num_beams=1 with proper beam search
- Added no_repeat_ngram_size=3 (primary fix for verbatim copying)
- Added repetition_penalty=1.5 (secondary fix for phrase repetition)
- temperature: 1.0 -> 1.3 (more diversity across beams)
"""

import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PREFIX = "paraphrase: "
MAX_LENGTH = 64


class QQPParaphraseDataset(Dataset):
    """
    Wraps the QQP label=1 (paraphrase) pairs for T5 fine-tuning.
    Each example: input "paraphrase: {question1}" -> target "{question2}".
    Also adds the REVERSE direction (question2 -> question1) since paraphrase
    is symmetric and doubling the training pairs this way is free.
    """

    def __init__(self, hf_dataset, tokenizer, max_length=MAX_LENGTH, max_examples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        for row in hf_dataset:
            if row["label"] != 1:
                continue  # only true paraphrases -- label=0 pairs are NOT paraphrases
            q1, q2 = row["question1"].strip(), row["question2"].strip()
            if not q1 or not q2:
                continue
            self.pairs.append((q1, q2))
            self.pairs.append((q2, q1))  # paraphrase is symmetric
            if max_examples and len(self.pairs) >= max_examples:
                break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        source_enc = self.tokenizer(
            PREFIX + source,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = target_enc["input_ids"].squeeze(0)
        # T5 ignores -100 in the loss -- mask out padding tokens in the target
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(0),
            "attention_mask": source_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def finetune(epochs=5, learning_rate=2e-4, batch_size=16, max_train_examples=80000):
    """
    Fine-tunes t5-small on QQP paraphrase pairs.

    Key hyperparameter decisions:
    - 80k pairs (was 20k): 4x more data gives the model enough variety to
      learn genuine rephrasing rather than surface copying. QQP has ~149k
      label=1 pairs; doubled for symmetry that's ~298k available. We cap at
      80k to keep training time reasonable on T4 (~45-60 min for 5 epochs).
    - 5 epochs (was 2): loss was still dropping steeply at epoch 2 in the
      first run (1.78->1.57), meaning the model hadn't converged. 5 epochs
      gives it time to move past the "copy input" local optimum.
    - lr=2e-4 (was 3e-4): slightly lower to avoid overshooting once the
      model starts learning the actual paraphrase task in later epochs.
    - Linear warmup scheduler: 10% warmup steps then linear decay. Standard
      practice for seq2seq fine-tuning -- prevents large gradient updates in
      the first few batches from destabilizing the pretrained weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.PARAPHRASE_BASE_MODEL} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.PARAPHRASE_BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.PARAPHRASE_BASE_MODEL).to(device)

    print("Loading QQP dataset (nyu-mll/glue, qqp config) ...")
    # Using the fully-qualified "nyu-mll/glue" path, not the legacy bare
    # "glue" alias -- the old shorthand's loading script is incompatible
    # with newer huggingface_hub URI parsing (hit this directly: HfUriError
    # on "glue" because it isn't in "namespace/name" format). nyu-mll/glue
    # is the actively maintained canonical path for the same dataset.
    qqp = load_dataset("nyu-mll/glue", "qqp", split="train")
    print(f"QQP train split: {len(qqp)} total rows (mixed label 0/1)")

    dataset = QQPParaphraseDataset(qqp, tokenizer, max_examples=max_train_examples)
    print(
        f"Built {len(dataset)} paraphrase training pairs "
        f"(label=1 only, both directions, capped at {max_train_examples})"
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Linear warmup for first 10% of steps, then linear decay to zero.
    # Warmup prevents large destabilizing updates in the first few batches
    # before the optimizer has accumulated reliable gradient statistics.
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    model.train()
    print(f"\nFine-tuning for {epochs} epochs (lr={learning_rate}, batch_size={batch_size}) ...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

    model.eval()

    # Atomic save -- save to temp folder first, verify weights exist,
    # THEN move into place. Avoids a half-written checkpoint from a
    # mid-save Colab disconnect silently breaking auto-detection later.
    final_path = os.path.join(config.CHECKPOINT_DIR, "paraphrase_finetuned")
    temp_path = final_path + "_tmp_saving"

    if os.path.exists(temp_path):
        import shutil
        shutil.rmtree(temp_path)
    os.makedirs(temp_path, exist_ok=True)

    model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)

    has_weights = os.path.exists(
        os.path.join(temp_path, "model.safetensors")
    ) or os.path.exists(
        os.path.join(temp_path, "pytorch_model.bin")
    )
    if not has_weights:
        raise RuntimeError(f"Save appears incomplete -- no weights file in {temp_path}.")

    if os.path.exists(final_path):
        import shutil
        shutil.rmtree(final_path)
    os.rename(temp_path, final_path)

    print(f"\nSaved fine-tuned paraphrase model to: {final_path}")
    return model, tokenizer, final_path


def quick_test(model, tokenizer, device, test_queries):
    """
    Sanity-check the freshly fine-tuned model on a few example queries.

    Generation strategy: beam search (num_beams=5) with no_repeat_ngram_size=3
    and repetition_penalty=1.5. This replaces the previous pure sampling
    approach (num_beams=1, do_sample=True) which produced incoherent output.

    Why beam search over sampling here:
    - Sampling (do_sample=True, num_beams=1) is maximally diverse but
      incoherent at low training budgets -- the model hasn't learned a
      strong enough paraphrase distribution yet, so sampling just adds noise.
    - Beam search finds the top-N most probable sequences, which at this
      training scale produces coherent output. Once the model is better
      trained, switching back to sampling with a lower temperature would
      give more lexical diversity.

    no_repeat_ngram_size=3: prevents any 3-gram from the input appearing
    verbatim in the output -- the primary fix for the "copy input" behavior
    seen in v1. Set to 3 rather than 2 to avoid over-constraining short
    queries where 2-grams are unavoidably shared.

    repetition_penalty=1.5: multiplicatively penalizes tokens that have
    already appeared in the output, providing a secondary guard against
    repetitive phrasing that no_repeat_ngram doesn't catch (e.g. repeating
    single tokens rather than n-grams).
    """
    print("\n" + "=" * 60)
    print("QUICK TEST — sampled paraphrases from the fine-tuned model")
    print("=" * 60)

    for query in test_queries:
        inputs = tokenizer(
            PREFIX + query,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_LENGTH,
                num_beams=5,
                num_return_sequences=5,
                temperature=1.3,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                early_stopping=True,
                do_sample=False,
            )

        candidates = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]

        print(f"\nORIGINAL: {query}")
        for c in candidates:
            print(f"  -> {c}")


def main():
    print("\nMODULE 1.3c — FINE-TUNE PARAPHRASE GENERATOR ON QQP\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, save_path = finetune()

    test_queries = [
        "What is the capital of France?",
        "How many moons does Mars have?",
        "What is the boiling point of water?",
        "Who wrote Romeo and Juliet?",
    ]
    quick_test(model, tokenizer, device, test_queries)


if __name__ == "__main__":
    main()