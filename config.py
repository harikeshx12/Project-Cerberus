"""
Central config for the project-cerberus project.
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
NLI_MODEL_NAME = "roberta-large-mnli"          # semantic equivalence classifier base
TRANSLATION_EN_FR = "Helsinki-NLP/opus-mt-en-fr"
TRANSLATION_FR_EN = "Helsinki-NLP/opus-mt-fr-en"
PARAPHRASE_BASE_MODEL = "t5-small"             # fine-tuned later on Quora Question Pairs

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
