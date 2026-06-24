"""
Module 1.2 — Semantic Equivalence Classifier

Goal: a reliable way to say "these two queries mean the same thing."
Everything downstream (variant generation, brittleness scoring) depends on
this being accurate, since it's the gate that decides whether a generated
variant actually preserved meaning or accidentally changed it.

Approach:
  1. Wrap roberta-large-mnli (general-purpose NLI model) in a clean class
     with a .score(query_a, query_b) method.
  2. Validate it against a hand-labeled test set (data/semantic_equivalence_test_pairs.json)
     and report accuracy.
  3. (Stub for now) fine-tuning hook for sharpening it on your specific task
     domain once you have enough labeled examples from real attack runs.

NLI models output 3 classes: entailment, neutral, contradiction.
We treat "entailment" (in either direction) as semantically equivalent,
and "contradiction" or "neutral" as not equivalent -- this is a simplification
worth revisiting once you see real failure cases.
"""

import os
import sys
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SemanticEquivalenceClassifier:
    """
    Wraps an NLI model to score whether two queries are semantically equivalent.

    Usage:
        clf = SemanticEquivalenceClassifier()
        result = clf.score("What is the capital of France?", "Which city is France's capital?")
        # result = {"label": "same", "confidence": 0.94, "raw_probs": {...}}
    """

    LABELS = ["contradiction", "neutral", "entailment"]  # roberta-large-mnli's output order

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or config.NLI_MODEL_NAME
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {self.model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _nli_probs(self, premise, hypothesis):
        """Run one direction of NLI and return softmax probs over the 3 classes."""
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        return {label: probs[i].item() for i, label in enumerate(self.LABELS)}

    def score(self, query_a, query_b):
        """
        Score whether query_a and query_b are semantically equivalent.

        We check entailment in BOTH directions (a->b and b->a) since NLI
        is directional -- "A dog is an animal" entails "something is an
        animal" but not the reverse. True paraphrases should entail in
        both directions; that's a stronger and more reliable signal than
        checking just one direction.
        """
        probs_ab = self._nli_probs(query_a, query_b)
        probs_ba = self._nli_probs(query_b, query_a)

        # Average entailment probability across both directions
        avg_entailment = (probs_ab["entailment"] + probs_ba["entailment"]) / 2
        avg_contradiction = (probs_ab["contradiction"] + probs_ba["contradiction"]) / 2

        if avg_entailment > 0.5 and avg_entailment > avg_contradiction:
            label = "same"
            confidence = avg_entailment
        else:
            label = "different"
            confidence = max(avg_contradiction, 1 - avg_entailment)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "raw_probs": {
                "a_to_b": {k: round(v, 4) for k, v in probs_ab.items()},
                "b_to_a": {k: round(v, 4) for k, v in probs_ba.items()},
            },
        }


def load_test_pairs(path=None):
    """Load the hand-labeled validation pairs."""
    path = path or os.path.join(config.DATA_DIR, "..", "data", "semantic_equivalence_test_pairs.json")
    # Fall back to repo-relative path if config.DATA_DIR points to Drive (Colab)
    # and the data file lives in the repo instead.
    repo_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "semantic_equivalence_test_pairs.json",
    )
    if os.path.exists(repo_data_path):
        path = repo_data_path

    with open(path, "r") as f:
        data = json.load(f)
    return data["pairs"]


def validate(classifier, pairs, verbose=True):
    """
    Run the classifier over all hand-labeled pairs and report accuracy.
    Returns (accuracy, list_of_misclassified_pairs) so you can inspect failures.
    """
    correct = 0
    misclassified = []

    for pair in pairs:
        result = classifier.score(pair["query_a"], pair["query_b"])
        is_correct = result["label"] == pair["label"]
        correct += int(is_correct)

        if verbose:
            mark = "OK " if is_correct else "FAIL"
            print(
                f"[{mark}] id={pair['id']:>2} expected={pair['label']:<9} "
                f"got={result['label']:<9} (conf={result['confidence']:.3f})  "
                f"'{pair['query_a']}' <-> '{pair['query_b']}'"
            )

        if not is_correct:
            misclassified.append({**pair, "predicted": result})

    accuracy = correct / len(pairs)
    return accuracy, misclassified


def main():
    print("\nMODULE 1.2 — SEMANTIC EQUIVALENCE CLASSIFIER\n")

    clf = SemanticEquivalenceClassifier()
    pairs = load_test_pairs()
    print(f"\nLoaded {len(pairs)} hand-labeled test pairs.\n")

    print("=" * 60)
    print("VALIDATION RUN")
    print("=" * 60)
    accuracy, misclassified = validate(clf, pairs)

    print()
    print("=" * 60)
    print(f"ACCURACY: {accuracy:.1%}  (threshold: {config.SEMANTIC_EQUIVALENCE_THRESHOLD:.0%})")
    print("=" * 60)

    if accuracy >= config.SEMANTIC_EQUIVALENCE_THRESHOLD:
        print("PASSED. Classifier meets the accuracy bar -- ready for Module 1.3.")
    else:
        print(f"BELOW THRESHOLD. {len(misclassified)} pairs misclassified:")
        for m in misclassified:
            print(f"  - id={m['id']}: expected '{m['label']}', got '{m['predicted']['label']}'")
        print("\nThis is expected on the FIRST run with the off-the-shelf model --")
        print("this is exactly why Module 1.2 includes a fine-tuning step. Review")
        print("the misclassified pairs above before deciding whether to fine-tune")
        print("or whether some labels need correcting.")

    # Save results to Drive for tracking across runs
    out_path = os.path.join(config.CHECKPOINT_DIR, "module1_2_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {"accuracy": accuracy, "num_pairs": len(pairs), "misclassified": misclassified},
            f,
            indent=2,
        )
    print(f"\nSaved validation results to: {out_path}")


if __name__ == "__main__":
    main()
