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

# Common irregular verbs likely to show up in factual/trivia-style queries
# this project targets ("Who wrote X", "Who discovered Y", "Who painted Z").
# Naive lemma+"ed" gets every one of these wrong (write->writed instead of
# written, see->seed instead of seen) -- exactly the bug found in testing.
# Not exhaustive; passive_voice() returns None (skips) for anything not
# covered here or by the regular-verb rule, rather than guessing.
IRREGULAR_PAST_PARTICIPLES = {
    "write": "written", "see": "seen", "give": "given", "take": "taken",
    "eat": "eaten", "break": "broken", "speak": "spoken", "wake": "woken",
    "drive": "driven", "ride": "ridden", "rise": "risen", "choose": "chosen",
    "steal": "stolen", "freeze": "frozen", "fall": "fallen", "forget": "forgotten",
    "discover": "discovered",  # regular, listed for clarity since it's common here
    "paint": "painted", "build": "built", "find": "found", "make": "made",
    "send": "sent", "bring": "brought", "buy": "bought",
    "catch": "caught", "teach": "taught", "think": "thought", "win": "won",
    "lose": "lost", "leave": "left", "feel": "felt", "tell": "told",
    "sell": "sold", "hold": "held", "lead": "led", "read": "read",
    "say": "said", "pay": "paid", "meet": "met", "sit": "sat", "sing": "sung",
    "begin": "begun", "drink": "drunk", "swim": "swum", "fly": "flown",
    "throw": "thrown", "draw": "drawn", "grow": "grown", "know": "known",
    "show": "shown", "blow": "blown",
}


def _past_participle(lemma):
    """
    Returns the past participle for a verb lemma, or None if unknown.
    Checks the irregular table first; falls back to regular -ed spelling
    rules (double consonant, drop-e, etc.) only for lemmas NOT in the
    irregular list -- since applying regular rules to an irregular verb
    is exactly how "wrote" -> "writed" happened.
    """
    lemma = lemma.lower()
    if lemma in IRREGULAR_PAST_PARTICIPLES:
        return IRREGULAR_PAST_PARTICIPLES[lemma]

    # Heuristic check: if this lemma looks like it's probably irregular
    # (common irregular verb endings) but isn't in our table, refuse to
    # guess rather than risk another "writed"-style error.
    likely_irregular_endings = ("ow", "ear", "ing", "ink", "ind")
    if any(lemma.endswith(suffix) for suffix in likely_irregular_endings):
        return None

    # Regular verb spelling rules
    if lemma.endswith("e"):
        return lemma + "d"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return lemma[:-1] + "ied"
    if len(lemma) >= 3 and lemma[-1] not in "aeiouwxy" and lemma[-2] in "aeiou" and lemma[-3] not in "aeiou":
        # short vowel + single final consonant -> double it (e.g. "chat" -> "chatted")
        # but skip this for longer/common verbs where it's more often wrong than right
        if len(lemma) <= 5:
            return lemma + lemma[-1] + "ed"
    return lemma + "ed"


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
            # "What is X" -> "Tell me X" (dropping "what is") is more natural
            # than "Tell me what X is". Use doc[2].idx (spaCy's real character
            # offset for the 3rd token) to slice, rather than manually summing
            # token text lengths -- that approach broke on whitespace and ate
            # part of the next word (produced "Tell me s the capital..." once).
            if len(doc) > 2:
                rest = stripped[doc[2].idx : -1].strip()
                return f"Tell me {rest}."
            return None
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

        # Find where to insert "not": if the adjective is preceded by a
        # determiner ("a good conductor"), "not" goes BEFORE the determiner
        # ("not a bad conductor"), not directly before the adjective --
        # "a not bad conductor" is grammatically wrong. Walk backward from
        # the adjective to find a contiguous det/advmod run to insert before.
        insert_idx = idx
        i = idx - 1
        while i >= 0 and doc[i].dep_ in ("det", "advmod") and doc[i].head.i == idx:
            insert_idx = i
            i -= 1

        tokens = [tok.text_with_ws for tok in doc]
        tokens[idx] = antonym + doc[idx].whitespace_
        tokens.insert(insert_idx, "not ")
        return "".join(tokens)

    def passive_voice(self, text):
        """
        Converts a simple Subject-Verb-Object DECLARATIVE sentence to passive
        voice. "The dog chased the cat." -> "The cat was chased by the dog."
        Only fires on sentences spaCy parses with a clean single nsubj +
        ROOT verb + dobj pattern -- anything more complex is skipped
        rather than risking a broken rewrite.

        Explicitly refuses to fire on questions: "passive voice of a
        question" isn't a coherent transformation (e.g. it previously
        turned "Who wrote Romeo and Juliet?" into "The Romeo was writed
        by Who." -- nonsensical, and also exposed the irregular-verb bug
        below). Declarative-only is the correct scope for this rule.
        """
        stripped = text.strip()
        if stripped.endswith("?"):
            return None  # passive voice doesn't apply to questions

        doc = self.nlp(stripped)

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

        # Build subject/object noun phrases (include determiners/compounds),
        # preserving each token's original capitalization and using the
        # ORIGINAL surface text, not a lowercased/recapitalized guess.
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

        participle = _past_participle(root.lemma_)
        if participle is None:
            return None  # unknown irregular verb -- skip rather than guess wrong

        be_verb = "were" if subject.tag_ == "NNS" else "was"

        # New subject is the object phrase, capitalized as a sentence start
        # (first letter upper, rest of the phrase untouched -- not blindly
        # re-capitalizing every word).
        new_subject = obj_phrase[0].upper() + obj_phrase[1:] if obj_phrase else obj_phrase
        # New "by" actor is the original subject phrase, lowercased ONLY if
        # it isn't a proper noun (check spaCy's own tag, not a guess based
        # on string position -- "The dog" -> "the dog" but "Who" stays "Who").
        if subject.pos_ == "PROPN" or subject.tag_ == "WP":
            by_phrase = subj_phrase
        else:
            by_phrase = subj_phrase[0].lower() + subj_phrase[1:] if subj_phrase else subj_phrase

        return f"{new_subject} {be_verb} {participle} by {by_phrase}."

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
