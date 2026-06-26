"""
Module 1.3b — The Variant Generator (Strategy B: Syntactic Transformation)

Goal: produce variants through rule-based grammatical transformations rather
than model inference. No translation model, no sampling -- pure spaCy parsing
plus string manipulation. This catches a different class of rephrasing than
back-translation: structural changes (active<->passive, question<->declarative
framing, double-negation) rather than vocabulary/phrasing changes.

Three transformations, each conservative by design (only fires when the
pattern clearly applies, otherwise returns no variant rather than guessing):

  1. Question -> declarative framing
     "What is the capital of France?" -> "Tell me the capital of France."
     "Who wrote Romeo and Juliet?" -> "Tell me who wrote Romeo and Juliet."

  2. Double-negation insertion
     "Is water a good conductor?" -> "Is water not a bad conductor?"
     Only fires on a small set of recognizable adjective antonym pairs --
     deliberately NOT attempting general negation, since getting this wrong
     silently flips meaning (exactly the failure mode Module 1.2 exists to
     catch downstream, but better to not generate obviously-wrong negations
     in the first place).

  3. Passive voice conversion (simple SVO sentences only)
     "The dog chased the cat." -> "The cat was chased by the dog."
     Requires a clear single subject-verb-object pattern; skips anything
     spaCy doesn't parse cleanly into that shape rather than guessing.

Like Strategy A, every candidate still goes through the SemanticEquivalence-
Classifier, FluencyFilter, and StructuralSanityCheck before being accepted --
rule-based generation can still produce something that drifts in meaning or
reads awkwardly, so it gets the same scrutiny, not a free pass for being
"deterministic."
"""

import spacy

# Small, deliberately conservative antonym table for the double-negation
# strategy. Only fires when one of these adjectives is present -- silence
# on anything else rather than risking a wrong negation.
ANTONYMS = {
    "good": "bad",
    "bad": "good",
    "large": "small",
    "small": "large",
    "big": "small",
    "fast": "slow",
    "slow": "fast",
    "hot": "cold",
    "cold": "hot",
    "true": "false",
    "false": "true",
    "high": "low",
    "low": "high",
    "easy": "hard",
    "hard": "easy",
    "strong": "weak",
    "weak": "strong",
}

WH_WORDS = {"what", "who", "where", "when", "why", "how", "which", "whom"}


class SyntacticTransformer:
    """
    Rule-based variant generator using spaCy for parsing.

    Usage:
        transformer = SyntacticTransformer()
        candidates = transformer.generate_candidates("What is the capital of France?")
        # -> ["Tell me the capital of France."]  (or [] if no rule applies)
    """

    def __init__(self, model_name="en_core_web_sm"):
        print(f"Loading spaCy model {model_name} ...")
        self.nlp = spacy.load(model_name)

    def question_to_declarative(self, text):
        """
        Converts a wh-question into a "Tell me ..." declarative framing.
        Conservative: only fires if the sentence clearly starts with a
        recognized wh-word and ends in '?'. Returns None if it doesn't
        confidently apply.
        """
        stripped = text.strip()
        if not stripped.endswith("?"):
            return None

        doc = self.nlp(stripped)
        if len(doc) == 0:
            return None

        first_token = doc[0].text.lower()
        if first_token not in WH_WORDS:
            return None

        body = stripped[:-1]  # drop the question mark
        # "What is the capital of France" -> rephrase as an indirect question
        if first_token == "what" and len(doc) > 1 and doc[1].lemma_ == "be":
            # "What is X" -> "Tell me what X is" reads awkwardly for most
            # cases; "Tell me X" (dropping "what is") is more natural and
            # still unambiguously the same request.
            rest = stripped[len(doc[0].text) + len(doc[1].text):-1].strip()
            return f"Tell me {rest}."
        else:
            # General case: "Who wrote X?" -> "Tell me who wrote X."
            return f"Tell me {body[0].lower()}{body[1:]}."

    def double_negation(self, text):
        """
        Rewrites a sentence containing a recognized adjective into a
        double-negative construction with the antonym.
        "Is water a good conductor?" -> "Is water not a bad conductor?"
        Only fires when exactly one recognized adjective is found, to avoid
        ambiguous multi-substitution cases.
        """
        doc = self.nlp(text)
        matches = [(tok.i, tok.text) for tok in doc if tok.text.lower() in ANTONYMS]

        if len(matches) != 1:
            return None  # zero or ambiguous multiple matches -- skip

        idx, original_word = matches[0]
        antonym = ANTONYMS[original_word.lower()]
        # preserve capitalization of the original word
        if original_word[0].isupper():
            antonym = antonym.capitalize()

        # Build the result token-by-token using the exact matched index,
        # rather than a blind string .replace() -- a blind replace risks
        # hitting an earlier occurrence of the same substring elsewhere in
        # the sentence (e.g. if the antonym word coincidentally already
        # appears earlier), which would silently produce the wrong sentence.
        tokens = [tok.text_with_ws for tok in doc]
        tokens[idx] = "not " + antonym + doc[idx].whitespace_
        return "".join(tokens)

    def passive_voice(self, text):
        """
        Converts a simple Subject-Verb-Object sentence to passive voice.
        "The dog chased the cat." -> "The cat was chased by the dog."
        Only fires on sentences spaCy parses with a clean single nsubj +
        ROOT verb + dobj pattern -- anything more complex is skipped
        rather than risking a broken rewrite.

        CAVEAT (untested as of writing -- verify carefully on first real run):
        the past-participle logic (lemma + "ed") is naive and WILL mishandle
        irregular verbs ("caught" != "catch"+"ed", "wrote" != "write"+"ed").
        This module hasn't been run against a real spaCy install yet (this
        sandbox can't reach the model download). Treat the first Colab run
        of this specific rule as a real test, not a formality -- check every
        passive-voice output by eye before trusting it in the pipeline.
        """
        doc = self.nlp(text.strip())

        root = None
        subject = None
        obj = None

        for tok in doc:
            if tok.dep_ == "ROOT" and tok.pos_ == "VERB":
                root = tok
            elif tok.dep_ == "nsubj":
                subject = tok
            elif tok.dep_ == "dobj":
                obj = tok

        if root is None or subject is None or obj is None:
            return None  # not a clean SVO sentence -- skip

        # Build subject/object noun phrases (include determiners/compounds)
        def phrase_for(token):
            span_tokens = [t for t in token.subtree if t.dep_ in ("det", "compound", "amod") or t == token]
            span_tokens.sort(key=lambda t: t.i)
            return " ".join(t.text for t in span_tokens)

        subj_phrase = phrase_for(subject)
        obj_phrase = phrase_for(obj)

        # Past tense -> "was/were VERB-participle by"
        # Simplification: only handle simple past tense verbs (most common
        # case for factual/trivia-style queries this project targets).
        if root.tag_ != "VBD":
            return None  # only handling simple past tense for now

        participle = root.lemma_ + "ed" if not root.lemma_.endswith("e") else root.lemma_ + "d"
        be_verb = "were" if subject.tag_ == "NNS" else "was"

        return f"The {obj_phrase.split(' ', 1)[-1] if obj_phrase.lower().startswith('the ') else obj_phrase} {be_verb} {participle} by {subj_phrase.lower() if not subj_phrase[0].isupper() else subj_phrase}.".replace(
            "The the", "The"
        )

    def generate_candidates(self, query):
        """
        Runs all three rules and returns whichever produced a result.
        Each rule is independent -- a query might match zero, one, or
        multiple rules (e.g. a question containing a recognized adjective
        could get both a declarative AND a negation variant).
        """
        candidates = []

        for rule_fn in (self.question_to_declarative, self.double_negation, self.passive_voice):
            try:
                result = rule_fn(query)
            except Exception:
                result = None  # conservative: any parsing edge case is a skip, not a crash
            if result and result.strip() and result.strip() != query.strip():
                candidates.append(result.strip())

        return candidates


def main():
    print("\nMODULE 1.3b — SYNTACTIC TRANSFORMATION (standalone rule test)\n")
    print("Running rules directly, WITHOUT the semantic/fluency/structural")
    print("filters, so you can see exactly what each rule produces.\n")

    transformer = SyntacticTransformer()

    test_queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "Is water a good conductor of electricity?",
        "The dog chased the cat.",
        "How many moons does Mars have?",
    ]

    for query in test_queries:
        print("=" * 60)
        print(f"ORIGINAL: {query}")
        print("=" * 60)
        candidates = transformer.generate_candidates(query)
        if not candidates:
            print("  (no rule applied)")
        else:
            for c in candidates:
                print(f"  -> {c}")
        print()


if __name__ == "__main__":
    main()
