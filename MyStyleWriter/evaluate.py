"""
==============================================================================
SCRIPT 3 — Evaluation (evaluate.py)
Project #27: MyStyle Writer — Hemingway Style Fine-Tuning
==============================================================================

This script:
  1. Loads the base Qwen2.5 model
  2. Loads the fine-tuned Qwen2.5 with LoRA adapter
  3. Runs both models on 10 identical test prompts
  4. Calculates metrics: Perplexity, BLEU, Avg Sentence Length, Vocab Diversity
  5. Prints a clean formatted comparison table
  6. Saves full results to evaluation_report.txt

Usage (run locally or on Colab):
    python evaluate.py

Prerequisites:
    - Run train.py first to create the hemingway_adapter/ folder
    - GPU recommended but not required (will use CPU if no GPU)
==============================================================================
"""

import os
import sys
import math
import json

# --- Block ALL Hugging Face network access ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import numpy as np
from datetime import datetime
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Local Model Path ---
# Use the HuggingFace cache directory where the model was previously downloaded.
_HF_CACHE_MODEL = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--Qwen--Qwen2.5-3B-Instruct"
)
_SNAPSHOT_DIR = os.path.join(_HF_CACHE_MODEL, "snapshots")
if os.path.isdir(_SNAPSHOT_DIR) and os.listdir(_SNAPSHOT_DIR):
    _rev = os.listdir(_SNAPSHOT_DIR)[0]
    MODEL_NAME = os.path.join(_SNAPSHOT_DIR, _rev)
else:
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(SCRIPT_DIR, "best_adapter")
DATASET_DIR = os.path.join(SCRIPT_DIR, "hemingway_dataset")
REPORT_FILE = os.path.join(SCRIPT_DIR, "evaluation_report.txt")

# Generation settings
MAX_NEW_TOKENS = 150      # Max tokens to generate per prompt
TEMPERATURE = 0.7         # Sampling temperature (lower = more focused)
TOP_P = 0.9               # Nucleus sampling threshold
DO_SAMPLE = True          # Use sampling instead of greedy

# Test prompts — diverse scenarios to evaluate style transfer
TEST_PROMPTS = [
    "The old man walked to the river.",
    "She sat alone in the cafe.",
]


# ============================================================================
# STEP 1: LOAD MODELS
# ============================================================================

def validate_local_model(model_path: str) -> bool:
    """
    Validate that the local model directory exists and contains required files.
    If missing or incomplete, print a clear error and return False.
    """
    print(f"  Validating local model at: {model_path}")

    if not os.path.isdir(model_path):
        print(f"\n❌ LOCAL MODEL NOT FOUND!")
        print(f"   Expected model directory: {model_path}")
        print(f"\n   Download the model ONCE with:")
        print(f"     python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')\"")
        return False

    files_in_dir = os.listdir(model_path)
    missing = []
    if "config.json" not in files_in_dir:
        missing.append("config.json")
    has_weights = any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in files_in_dir if not f.endswith(".incomplete")
    )
    if not has_weights:
        missing.append("model weights (*.safetensors or *.bin)")
    has_tokenizer = "tokenizer.json" in files_in_dir or "tokenizer_config.json" in files_in_dir
    if not has_tokenizer:
        missing.append("tokenizer files")

    if missing:
        print(f"\n❌ LOCAL MODEL IS INCOMPLETE!")
        print(f"   Missing: {', '.join(missing)}")
        print(f"   Download the model ONCE with:")
        print(f"     python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')\"")
        return False

    print(f"  ✓ Local model validated")
    return True


def load_base_model(model_name: str):
    """
    Load the base model (without any fine-tuning) from LOCAL cache only.
    """
    print(f"  Loading base model from local cache...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

    model.eval()
    print(f"  ✓ Base model loaded on {device} (local)")
    return model, tokenizer, device


def load_finetuned_model(model_name: str, adapter_dir: str):
    """
    Load the fine-tuned model with the LoRA adapter applied (LOCAL only).
    """
    print(f"  Loading fine-tuned model with LoRA adapter (local)...")

    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(
            f"Adapter not found at '{adapter_dir}'. Run train.py first!"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )

    # Apply the LoRA adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
    model.eval()
    print(f"  ✓ Fine-tuned model loaded on {device} (local)")
    return model, tokenizer, device


# ============================================================================
# STEP 2: GENERATE TEXT
# ============================================================================

def generate_text(model, tokenizer, prompt: str, device: str) -> str:
    """
    Generate text from a given prompt using the model.
    Returns the generated continuation (excluding the prompt).
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        # Decode only the generated tokens (not the prompt)
        generated_tokens = output_ids[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text.strip()

    except Exception as e:
        return f"[Generation Error: {e}]"


# ============================================================================
# STEP 3: CALCULATE METRICS
# ============================================================================

def calculate_perplexity(model, tokenizer, text: str) -> float:
    """
    Calculate the perplexity of the model on a given text.
    Lower perplexity = model is more confident about the text.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

        perplexity = math.exp(loss.item())
        return min(perplexity, 10000.0)  # Cap at 10000 to avoid overflow display

    except Exception:
        return float("nan")


def calculate_bleu(generated: str, reference: str) -> float:
    """
    Calculate a simplified BLEU-like score between generated and reference text.
    Uses 1-gram to 4-gram overlap.
    """
    try:
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()

        if not gen_tokens or not ref_tokens:
            return 0.0

        # Calculate n-gram precisions for n=1 to 4
        precisions = []
        for n in range(1, 5):
            gen_ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]

            if not gen_ngrams:
                precisions.append(0.0)
                continue

            ref_counter = Counter(ref_ngrams)
            matches = 0
            for ngram in gen_ngrams:
                if ref_counter[ngram] > 0:
                    matches += 1
                    ref_counter[ngram] -= 1

            precision = matches / len(gen_ngrams)
            precisions.append(precision)

        # Geometric mean of precisions (with smoothing)
        smoothed = [max(p, 1e-10) for p in precisions]
        log_avg = sum(math.log(p) for p in smoothed) / len(smoothed)
        bleu = math.exp(log_avg)

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(gen_tokens), 1)))
        return bleu * bp

    except Exception:
        return 0.0


def calculate_avg_sentence_length(text: str) -> float:
    """
    Calculate average sentence length in words.
    Hemingway style = short sentences (typically 5-15 words).
    """
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(word_counts)


def calculate_vocab_diversity(text: str) -> float:
    """
    Calculate vocabulary diversity: unique words / total words.
    Higher = more diverse vocabulary. Hemingway tends to be lower (simple words).
    """
    words = text.lower().split()
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)


# ============================================================================
# STEP 4: RUN EVALUATION
# ============================================================================

def run_evaluation():
    """
    Main evaluation pipeline:
    1. Load both models
    2. Generate from 10 prompts
    3. Calculate metrics
    4. Print comparison table
    5. Save report
    """
    print("\n" + "★" * 60)
    print("  MyStyle Writer — Model Evaluation")
    print("  Comparing: Base Qwen2.5 vs Fine-tuned (LOCAL models only)")
    print("★" * 60 + "\n")

    # --- Validate local model before loading ---
    if not validate_local_model(MODEL_NAME):
        print("\n❌ Cannot proceed without the local model. Exiting.")
        sys.exit(1)

    # --- Load reference texts for BLEU calculation ---
    reference_texts = []
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(DATASET_DIR)
        test_texts = dataset["test"]["text"]
        reference_texts = test_texts[:10] if len(test_texts) >= 10 else test_texts
        print(f"  Loaded {len(reference_texts)} reference texts for BLEU scoring\n")
    except Exception as e:
        print(f"  ⚠ Could not load reference texts: {e}")
        print(f"  BLEU scores will use prompts as approximate references.\n")

    # --- Step 1: Load base model ---
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    base_model, base_tokenizer, device = load_base_model(MODEL_NAME)

    # --- Step 2: Load fine-tuned model ---
    try:
        ft_model, ft_tokenizer, _ = load_finetuned_model(MODEL_NAME, ADAPTER_DIR)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("   Cannot run evaluation without the fine-tuned adapter.")
        return

    # --- Step 3: Generate and evaluate ---
    print("\n" + "=" * 60)
    print("Generating outputs and calculating metrics...")
    print("=" * 60)

    results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n  Prompt {i+1}/{len(TEST_PROMPTS)}: \"{prompt[:50]}...\"")

        # Generate from base model
        base_output = generate_text(base_model, base_tokenizer, prompt, device)
        print(f"    Base output:      {base_output[:60]}...")

        # Generate from fine-tuned model
        ft_output = generate_text(ft_model, ft_tokenizer, prompt, device)
        print(f"    Fine-tuned output: {ft_output[:60]}...")

        # Use reference text for BLEU (or fall back to prompt)
        reference = reference_texts[i] if i < len(reference_texts) else prompt

        # Calculate metrics for base model output
        base_perplexity = calculate_perplexity(base_model, base_tokenizer, prompt + " " + base_output)
        base_bleu = calculate_bleu(base_output, reference)
        base_avg_sent_len = calculate_avg_sentence_length(base_output)
        base_vocab_div = calculate_vocab_diversity(base_output)

        # Calculate metrics for fine-tuned model output
        ft_perplexity = calculate_perplexity(ft_model, ft_tokenizer, prompt + " " + ft_output)
        ft_bleu = calculate_bleu(ft_output, reference)
        ft_avg_sent_len = calculate_avg_sentence_length(ft_output)
        ft_vocab_div = calculate_vocab_diversity(ft_output)

        result = {
            "prompt": prompt,
            "base_output": base_output,
            "ft_output": ft_output,
            "base_metrics": {
                "perplexity": base_perplexity,
                "bleu": base_bleu,
                "avg_sentence_length": base_avg_sent_len,
                "vocab_diversity": base_vocab_div,
            },
            "ft_metrics": {
                "perplexity": ft_perplexity,
                "bleu": ft_bleu,
                "avg_sentence_length": ft_avg_sent_len,
                "vocab_diversity": ft_vocab_div,
            },
        }
        results.append(result)

    # --- Step 4: Print comparison table ---
    print_comparison_table(results)

    # --- Step 5: Save report ---
    save_report(results)

    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print(f"   Full report saved to: {REPORT_FILE}")
    print("   Run app.py to launch the interactive Gradio UI.")
    print("=" * 60 + "\n")


# ============================================================================
# STEP 5: PRINT COMPARISON TABLE
# ============================================================================

def print_comparison_table(results: list):
    """
    Print a clean formatted comparison table in the terminal.
    Shows average metrics for base vs fine-tuned model.
    """
    print("\n\n" + "=" * 80)
    print(" " * 15 + "EVALUATION RESULTS — COMPARISON TABLE")
    print("=" * 80)

    # Calculate averages
    base_metrics = {
        "perplexity": [],
        "bleu": [],
        "avg_sentence_length": [],
        "vocab_diversity": [],
    }
    ft_metrics = {
        "perplexity": [],
        "bleu": [],
        "avg_sentence_length": [],
        "vocab_diversity": [],
    }

    for r in results:
        for key in base_metrics:
            val = r["base_metrics"][key]
            if not math.isnan(val):
                base_metrics[key].append(val)
            val = r["ft_metrics"][key]
            if not math.isnan(val):
                ft_metrics[key].append(val)

    # Print header
    print(f"\n{'Metric':<25} {'Base Model':>15} {'Fine-tuned':>15} {'Better?':>10}")
    print("-" * 65)

    # Perplexity (lower is better)
    base_ppl = np.mean(base_metrics["perplexity"]) if base_metrics["perplexity"] else float("nan")
    ft_ppl = np.mean(ft_metrics["perplexity"]) if ft_metrics["perplexity"] else float("nan")
    better_ppl = "Fine-tuned" if ft_ppl < base_ppl else "Base"
    print(f"{'Perplexity ↓':<25} {base_ppl:>15.2f} {ft_ppl:>15.2f} {better_ppl:>10}")

    # BLEU (higher is better)
    base_bleu = np.mean(base_metrics["bleu"]) if base_metrics["bleu"] else 0.0
    ft_bleu = np.mean(ft_metrics["bleu"]) if ft_metrics["bleu"] else 0.0
    better_bleu = "Fine-tuned" if ft_bleu > base_bleu else "Base"
    print(f"{'BLEU Score ↑':<25} {base_bleu:>15.4f} {ft_bleu:>15.4f} {better_bleu:>10}")

    # Avg Sentence Length (shorter = more Hemingway)
    base_asl = np.mean(base_metrics["avg_sentence_length"]) if base_metrics["avg_sentence_length"] else 0.0
    ft_asl = np.mean(ft_metrics["avg_sentence_length"]) if ft_metrics["avg_sentence_length"] else 0.0
    better_asl = "Fine-tuned" if ft_asl < base_asl else "Base"
    print(f"{'Avg Sentence Len ↓':<25} {base_asl:>15.1f} {ft_asl:>15.1f} {better_asl:>10}")

    # Vocab Diversity (Hemingway = simpler, but not too low)
    base_vd = np.mean(base_metrics["vocab_diversity"]) if base_metrics["vocab_diversity"] else 0.0
    ft_vd = np.mean(ft_metrics["vocab_diversity"]) if ft_metrics["vocab_diversity"] else 0.0
    print(f"{'Vocab Diversity':<25} {base_vd:>15.2%} {ft_vd:>15.2%} {'—':>10}")

    print("-" * 65)

    # Style score: how Hemingway-like is the fine-tuned model?
    # Short sentences + lower perplexity + higher BLEU = better style match
    hemingway_target_asl = 8.0  # Hemingway's typical avg sentence length
    base_style = max(0, 100 - abs(base_asl - hemingway_target_asl) * 5)
    ft_style = max(0, 100 - abs(ft_asl - hemingway_target_asl) * 5)

    print(f"\n{'Style Match Score':<25} {base_style:>14.0f}% {ft_style:>14.0f}%")
    print("=" * 80)


# ============================================================================
# STEP 6: SAVE FULL REPORT
# ============================================================================

def save_report(results: list):
    """
    Save the complete evaluation report to a text file.
    Includes all prompts, outputs, and metrics.
    """
    print(f"\n  Saving evaluation report to: {REPORT_FILE}")

    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("  MyStyle Writer — Evaluation Report\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Base Model: {MODEL_NAME}\n")
            f.write(f"  Adapter: {ADAPTER_DIR}\n")
            f.write("=" * 80 + "\n\n")

            for i, r in enumerate(results):
                f.write(f"{'─' * 80}\n")
                f.write(f"  PROMPT {i+1}: {r['prompt']}\n")
                f.write(f"{'─' * 80}\n\n")

                f.write(f"  BASE MODEL OUTPUT:\n")
                f.write(f"  {r['base_output']}\n\n")

                f.write(f"  FINE-TUNED OUTPUT:\n")
                f.write(f"  {r['ft_output']}\n\n")

                f.write(f"  METRICS COMPARISON:\n")
                f.write(f"    {'Metric':<25} {'Base':>12} {'Fine-tuned':>12}\n")
                f.write(f"    {'-' * 49}\n")

                bm = r["base_metrics"]
                fm = r["ft_metrics"]

                f.write(f"    {'Perplexity':<25} {bm['perplexity']:>12.2f} {fm['perplexity']:>12.2f}\n")
                f.write(f"    {'BLEU Score':<25} {bm['bleu']:>12.4f} {fm['bleu']:>12.4f}\n")
                f.write(f"    {'Avg Sentence Length':<25} {bm['avg_sentence_length']:>12.1f} {fm['avg_sentence_length']:>12.1f}\n")
                f.write(f"    {'Vocab Diversity':<25} {bm['vocab_diversity']:>11.2%} {fm['vocab_diversity']:>11.2%}\n")
                f.write(f"\n\n")

            # Summary section
            f.write("=" * 80 + "\n")
            f.write("  SUMMARY — AVERAGE METRICS\n")
            f.write("=" * 80 + "\n\n")

            avg_base = {}
            avg_ft = {}
            for key in ["perplexity", "bleu", "avg_sentence_length", "vocab_diversity"]:
                base_vals = [r["base_metrics"][key] for r in results if not math.isnan(r["base_metrics"][key])]
                ft_vals = [r["ft_metrics"][key] for r in results if not math.isnan(r["ft_metrics"][key])]
                avg_base[key] = np.mean(base_vals) if base_vals else 0
                avg_ft[key] = np.mean(ft_vals) if ft_vals else 0

            f.write(f"    {'Metric':<25} {'Base':>12} {'Fine-tuned':>12}\n")
            f.write(f"    {'-' * 49}\n")
            f.write(f"    {'Perplexity':<25} {avg_base['perplexity']:>12.2f} {avg_ft['perplexity']:>12.2f}\n")
            f.write(f"    {'BLEU Score':<25} {avg_base['bleu']:>12.4f} {avg_ft['bleu']:>12.4f}\n")
            f.write(f"    {'Avg Sentence Length':<25} {avg_base['avg_sentence_length']:>12.1f} {avg_ft['avg_sentence_length']:>12.1f}\n")
            f.write(f"    {'Vocab Diversity':<25} {avg_base['vocab_diversity']:>11.2%} {avg_ft['vocab_diversity']:>11.2%}\n")

            f.write(f"\n\n  Report generated by MyStyle Writer evaluation pipeline.\n")

        print(f"  ✓ Report saved successfully!")

    except Exception as e:
        print(f"  ✗ Error saving report: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        run_evaluation()
    except FileNotFoundError as e:
        print(f"\n❌ FILE ERROR: {e}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ GPU OUT OF MEMORY!")
            print("   Try closing other applications or using a T4 GPU on Colab.")
        else:
            print(f"\n❌ RUNTIME ERROR: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise
