"""
Module 1.1 — Environment Setup

Goal: confirm the environment is correctly configured before any real
ML code gets built on top of it. This module:
  1. Checks GPU availability and reports what's allocated (T4 vs A100 vs CPU)
  2. Loads the base defender model (Flan-T5) and runs a test inference
  3. Saves a small test artifact to Drive to confirm the save path works
  4. Initializes a Weights & Biases run for experiment tracking

Run this first, every fresh Colab session, before touching Module 1.2+.
"""

import os
import sys
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Make sure project root is importable whether this is run as a script
# or imported as a module from a notebook.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def check_gpu():
    """Report GPU availability and type. Returns the torch device to use."""
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU available: {gpu_name}")
        print(f"Total memory: {total_mem_gb:.1f} GB")

        if "A100" in gpu_name:
            print("-> A100 detected. Good for Phase 3+ (training loop).")
        elif "T4" in gpu_name:
            print("-> T4 detected. Fine for Phase 1 inference work.")
        else:
            print(f"-> {gpu_name} detected. Should be workable for Phase 1.")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected. Running on CPU.")
        print("This is fine for testing this module, but inference will be")
        print("too slow (30-60s/query) for the full attack loop later.")
        print("Make sure Runtime > Change runtime type > GPU is set in Colab.")

    print()
    return device


def load_and_test_model(device):
    """Load Flan-T5-base and run one test inference to confirm everything works."""
    print("=" * 60)
    print("MODEL LOAD + TEST INFERENCE")
    print("=" * 60)

    print(f"Loading {config.BASE_MODEL_NAME} ...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.BASE_MODEL_NAME).to(device)
    load_time = time.time() - start
    print(f"Loaded in {load_time:.1f}s")

    test_query = "What is the capital of France?"
    print(f"\nTest query: {test_query!r}")

    inputs = tokenizer(test_query, return_tensors="pt").to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            output_scores=True,
            return_dict_in_generate=True,
        )
    inference_time = time.time() - start

    answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print(f"Answer: {answer!r}")
    print(f"Inference time: {inference_time:.2f}s")

    if inference_time > 5 and device.type == "cpu":
        print("\nNote: slow inference confirms CPU is not viable for the full")
        print("attack loop (thousands of queries). GPU is required from here on.")

    print()
    return tokenizer, model, {
        "test_query": test_query,
        "answer": answer,
        "load_time_sec": load_time,
        "inference_time_sec": inference_time,
        "device": str(device),
    }


def save_test_artifact(result):
    """Save a small JSON artifact to Drive to confirm the save path works."""
    print("=" * 60)
    print("DRIVE SAVE CHECK")
    print("=" * 60)

    out_path = os.path.join(config.CHECKPOINT_DIR, "module1_1_test_artifact.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved test artifact to: {out_path}")
    print(f"(IS_COLAB={config.IS_COLAB} -> this should be under your Drive folder")
    print(f" if running in Colab, or a local drive_mirror/ folder if running locally)")
    print()


def init_wandb():
    """Initialize a W&B run. Skips gracefully if wandb isn't logged in yet."""
    print("=" * 60)
    print("WEIGHTS & BIASES CHECK")
    print("=" * 60)

    try:
        import wandb
        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name="module1_1_setup_check",
            job_type="setup",
        )
        wandb.log({"setup_check": 1})
        run.finish()
        print("W&B run logged successfully.")
    except Exception as e:
        print(f"W&B not fully configured yet ({e}).")
        print("Run `wandb login` (paste your API key from wandb.ai/authorize)")
        print("the first time you use this in a fresh Colab session.")
    print()


def main():
    print("\nMODULE 1.1 — ENVIRONMENT SETUP\n")

    device = check_gpu()
    tokenizer, model, result = load_and_test_model(device)
    save_test_artifact(result)
    init_wandb()

    print("=" * 60)
    print("MODULE 1.1 COMPLETE")
    print("=" * 60)
    print("If everything above ran without errors, you're ready for Module 1.2")
    print("(the Semantic Equivalence Classifier).")


if __name__ == "__main__":
    main()
