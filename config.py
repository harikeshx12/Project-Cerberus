"""
Central config for the adversarial-triad project.
All paths, model names, and hyperparameters live here so nothing is hardcoded
inside individual modules.
"""

import os

# ---- Environment detection ----
# When running inside Colab, DRIVE_ROOT will exist after drive.mount().
# When running locally in VSCode, we fall back to a local ./drive_mirror folder
# so the same code paths work in both places without edits.
IS_COLAB = os.path.exists("/content")

if IS_COLAB:
    DRIVE_ROOT = "/content/drive/MyDrive/project-cerberus"
else:
    DRIVE_ROOT = os.path.join(os.path.dirname(__file__), "drive_mirror")

os.makedirs(DRIVE_ROOT, exist_ok=True)

# ---- Subdirectories on Drive (checkpoints, logs, datasets) ----
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")
LOG_DIR = os.path.join(DRIVE_ROOT, "attack_logs")
DATA_DIR = os.path.join(DRIVE_ROOT, "data")

for d in (CHECKPOINT_DIR, LOG_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

# ---- Models ----
BASE_MODEL_NAME = "google/flan-t5-base"        # the defender model we're attacking

# NLI_MODEL_NAME points to the fine-tuned Semantic Equivalence Classifier
# (patched on numeric semantics in Module 1.2b) if a COMPLETE checkpoint
# exists on Drive, otherwise falls back to the raw off-the-shelf model.
#
# Important: we check for the actual weight file, not just the folder --
# a folder can exist with only config.json/tokenizer files if a previous
# save_pretrained() got interrupted (e.g. Colab disconnect mid-write).
# Checking folder existence alone caused exactly that failure once already:
# a half-written checkpoint permanently broke auto-detection until manually
# deleted. This is now self-healing instead.
_FINETUNED_NLI_PATH = os.path.join(CHECKPOINT_DIR, "semantic_equivalence_finetuned")


def _is_complete_checkpoint(path):
    if not os.path.isdir(path):
        return False
    has_weights = os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(
        os.path.join(path, "pytorch_model.bin")
    )
    has_config = os.path.exists(os.path.join(path, "config.json"))
    return has_weights and has_config


NLI_MODEL_NAME = _FINETUNED_NLI_PATH if _is_complete_checkpoint(_FINETUNED_NLI_PATH) else "roberta-large-mnli"

TRANSLATION_EN_FR = "Helsinki-NLP/opus-mt-en-fr"
TRANSLATION_FR_EN = "Helsinki-NLP/opus-mt-fr-en"

# PARAPHRASE_MODEL_NAME points to the fine-tuned paraphrase generator
# (Module 1.3c, trained on QQP) if a complete checkpoint exists, otherwise
# falls back to the raw t5-small base (which hasn't learned to paraphrase
# specifically, but still loads and runs).
_FINETUNED_PARAPHRASE_PATH = os.path.join(CHECKPOINT_DIR, "paraphrase_finetuned")
PARAPHRASE_BASE_MODEL = "t5-small"
PARAPHRASE_MODEL_NAME = (
    _FINETUNED_PARAPHRASE_PATH if _is_complete_checkpoint(_FINETUNED_PARAPHRASE_PATH) else PARAPHRASE_BASE_MODEL
)

# ---- Weights & Biases ----
WANDB_PROJECT = "project-cerberus"
WANDB_ENTITY = None  # set to your wandb username/team if needed

# ---- Module 1.2 — Semantic equivalence classifier ----
SEMANTIC_EQUIVALENCE_THRESHOLD = 0.9  # min accuracy required on hand-labeled test set

# ---- Module 1.3 — Variant generator ----
NUM_VARIANTS_PER_QUERY = 10

# ---- Module 1.4 — Ground truth dataset ----
GROUND_TRUTH_SIZE = 200

# ---- Random seed for reproducibility ----
SEED = 42
