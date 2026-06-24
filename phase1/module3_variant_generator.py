"""
Module 1.3 — The Variant Generator (Strategy A: Back-translation)

Goal: given any input query, produce N semantically equivalent variants by
round-tripping through another language (English -> French -> English).
The pivot through French naturally produces fluent rephrasing -- different
surface form, same meaning -- without any rule-writing.

Diversity comes from sampling multiple candidates on the French->English leg
(temperature + multiple return sequences), rather than just doing one greedy
round-trip, since a single greedy back-translation tends to just give you
back the original sentence with trivial changes.

Every candidate variant is filtered through the SemanticEquivalenceClassifier
from Module 1.2 before being returned -- a back-translation that drifts in
meaning gets dropped, not silently included.

This module is intentionally structured so Strategy B (syntactic transforms)
and Strategy C (controlled paraphrase) can be added as additional methods
on VariantGenerator later, each feeding into the same generate() pipeline,
without needing to change how callers use this class.
"""

import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1.module2_semantic_equivalence import SemanticEquivalenceClassifier


class BackTranslationStrategy:
    """English -> French -> English round-trip, sampling multiple candidates
    on the return leg for diversity."""

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading translation models on {self.device} ...")
        self.tok_en_fr = AutoTokenizer.from_pretrained(config.TRANSLATION_EN_FR)
        self.model_en_fr = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATION_EN_FR).to(self.device)

        self.tok_fr_en = AutoTokenizer.from_pretrained(config.TRANSLATION_FR_EN)
        self.model_fr_en = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATION_FR_EN).to(self.device)

        self.model_en_fr.eval()
        self.model_fr_en.eval()

    def _translate_to_french(self, text):
        """Single, deterministic translation to French (the pivot step)."""
        inputs = self.tok_en_fr(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model_en_fr.generate(
                **inputs, max_new_tokens=64, max_length=None, num_beams=4
            )
        return self.tok_en_fr.decode(output_ids[0], skip_special_tokens=True)

    def _translate_to_english_variants(self, french_text, n=10):
        """
        Generate up to n diverse English candidates from the French pivot
        by sampling instead of greedy decoding. Some duplicates are expected
        and removed downstream.
        """
        inputs = self.tok_fr_en(french_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model_fr_en.generate(
                **inputs,
                max_new_tokens=64,
                max_length=None,
                do_sample=True,
                temperature=1.2,
                top_k=50,
                num_beams=1,  # override the model's default num_beams=4 --
                              # pure sampling doesn't need beam search, and
                              # leaving the default in place conflicts with
                              # num_return_sequences > num_beams
                num_return_sequences=n,
            )
        candidates = [
            self.tok_fr_en.decode(ids, skip_special_tokens=True) for ids in output_ids
        ]
        return candidates

    def generate_candidates(self, query, n=10):
        """Returns up to n raw candidate variants (unfiltered -- caller checks
        semantic equivalence)."""
        french = self._translate_to_french(query)
        candidates = self._translate_to_english_variants(french, n=n)
        return candidates


class VariantGenerator:
    """
    Orchestrates variant-generation strategies and filters every candidate
    through the SemanticEquivalenceClassifier before returning it.

    Usage:
        gen = VariantGenerator()
        variants = gen.generate("What is the capital of France?", n=10)
        # -> [{"text": "...", "confidence": 0.94, "strategy": "back_translation"}, ...]
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.back_translation = BackTranslationStrategy(device=self.device)
        self.equivalence_classifier = SemanticEquivalenceClassifier()
        # Strategy B (syntactic) and Strategy C (paraphrase) will be
        # initialized here once added.

    def generate(self, query, n=10, min_confidence=0.5):
        """
        Generate up to n semantically-equivalent variants of query.

        Over-generates raw candidates (2x n) since some will get filtered
        out for drifting in meaning or being exact duplicates of the
        original, then returns the top n by classifier confidence.
        """
        raw_candidates = self.back_translation.generate_candidates(query, n=n * 2)

        seen = {query.strip().lower()}
        scored_variants = []

        for candidate in raw_candidates:
            normalized = candidate.strip().lower()
            if not candidate.strip() or normalized in seen:
                continue
            seen.add(normalized)

            result = self.equivalence_classifier.score(query, candidate)
            if result["label"] == "same" and result["confidence"] >= min_confidence:
                scored_variants.append({
                    "text": candidate,
                    "confidence": result["confidence"],
                    "strategy": "back_translation",
                })

        scored_variants.sort(key=lambda v: v["confidence"], reverse=True)
        return scored_variants[:n]


def main():
    print("\nMODULE 1.3 — VARIANT GENERATOR (Back-translation)\n")

    gen = VariantGenerator()

    test_queries = [
        "What is the capital of France?",
        "How many moons does Mars have?",
        "What is the boiling point of water?",
        "Who wrote Romeo and Juliet?",
    ]

    for query in test_queries:
        print("=" * 60)
        print(f"ORIGINAL: {query}")
        print("=" * 60)

        variants = gen.generate(query, n=10)

        if not variants:
            print("  No variants passed the semantic equivalence filter.")
        else:
            for i, v in enumerate(variants, 1):
                print(f"  {i:>2}. (conf={v['confidence']:.3f}) {v['text']}")
        print()


if __name__ == "__main__":
    main()
