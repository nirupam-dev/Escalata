"""
==============================================================================
SCRIPT 2 — LoRA Training (train.py)
Project #27: MyStyle Writer — Style Fine-Tuning
==============================================================================

This script:
  1. Loads Qwen2.5-3B-Instruct from HuggingFace
  2. Configures LoRA adapters via PEFT (r=16, alpha=32)
  3. Loads databricks/databricks-dolly-15k from HuggingFace Hub
  4. Converts to Alpaca instruction format, cleans & deduplicates
  5. Splits into 80% train / 20% validation
  6. Trains using SFTTrainer from TRL — optimized for Colab free T4
  7. Reports training loss, validation loss, and perplexity
  8. Saves LoRA adapter weights to ./dolly_adapter/

Usage (run on Google Colab with T4 GPU):
    python train.py

Prerequisites:
    - Requires GPU (T4 recommended, ~15GB VRAM)
    - Internet connection to download dataset on first run
==============================================================================
"""

import os
import sys
import math
import random
import hashlib

# --- Block ALL Hugging Face network access ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from collections import Counter
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
try:
    from trl import SFTTrainer, SFTConfig
    HAS_SFT_CONFIG = True
except ImportError:
    from trl import SFTTrainer
    HAS_SFT_CONFIG = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Model Configuration ---
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

# --- Dataset Configuration ---
DATASET_NAME = "databricks/databricks-dolly-15k"  # HuggingFace Hub dataset
DATASET_SAMPLE_SIZE = 2000                         # Select 2000 samples (~1hr on T4)
DATASET_SEED = 42                                  # Seed for shuffle & split
VALIDATION_SPLIT = 0.2                             # 20% validation split

# --- Output Paths ---
# NOTE: __file__ is not defined in Colab notebooks, so we fall back to
# the current working directory (which is /content in Colab).
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()  # Colab fallback → /content
ADAPTER_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "best_adapter")

# --- LoRA Configuration (optimized for Colab free T4) ---
LORA_R = 16                          # Rank of the low-rank matrices
LORA_ALPHA = 32                      # Scaling factor (alpha/r = scaling)
LORA_DROPOUT = 0.05                  # Dropout probability for LoRA layers
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Which attention layers to adapt
LORA_BIAS = "none"                   # No bias adaptation

# --- Training Hyperparameters (optimized for T4 with 15GB VRAM) ---
NUM_TRAIN_EPOCHS = 1                 # Train for exactly 1 epoch over the dataset
BATCH_SIZE = 2                       # Per-device batch size
GRADIENT_ACCUMULATION = 4            # Effective batch = 2 * 4 = 8
LEARNING_RATE = 2e-4                 # Learning rate with AdamW
FP16_ENABLED = False                 # Disabled AMP to prevent T4 bfloat16 scaler crashes
SAVE_STEPS = 50                      # Save checkpoint every 50 steps
LOGGING_STEPS = 10                   # Log training loss every 10 steps
EVAL_STEPS = 50                      # Run validation every 50 steps
MAX_SEQ_LENGTH = 512                 # Maximum sequence length for training
WARMUP_RATIO = 0.05                  # Linear warmup ratio (5% of total steps)


# ============================================================================
# STEP 1: CHECK GPU AVAILABILITY
# ============================================================================

def check_gpu():
    """
    Verify that a GPU is available and print device info.
    Training on CPU is possible but extremely slow.
    """
    print("=" * 60)
    print("STEP 1: Checking GPU availability...")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU detected: {gpu_name}")
        print(f"  ✓ GPU memory:   {gpu_memory:.1f} GB")
        device = "cuda"
    else:
        print("  ⚠ No GPU detected! Training will be very slow on CPU.")
        print("  ⚠ Recommendation: Use Google Colab with T4 GPU.")
        device = "cpu"

    return device


# ============================================================================
# STEP 2: LOAD TOKENIZER AND MODEL
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


def load_model_and_tokenizer(model_name: str, device: str):
    """
    Load the base model and tokenizer from LOCAL cache only.
    Uses 4-bit quantization to fit in T4 GPU memory.
    """
    print("\n" + "=" * 60)
    print(f"STEP 2: Loading model and tokenizer (LOCAL only)...")
    print("=" * 60)

    # --- Load Tokenizer (local only) ---
    print("  Loading tokenizer from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True
    )

    # Set padding token (Some models don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # IMPORTANT: Causal LM training requires right-padding so the loss
    # is computed on the actual tokens, not the padding on the left.
    tokenizer.padding_side = "right"
    print(f"  ✓ Tokenizer loaded — vocab size: {tokenizer.vocab_size:,}")

    # --- 4-bit Quantization Config (saves GPU memory) ---
    print("  Configuring 4-bit quantization for memory efficiency...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch.float16, # Compute in fp16
        bnb_4bit_use_double_quant=True,       # Double quantization for extra savings
    )

    # --- Load Model (local only, no downloads) ---
    print(f"  Loading model from local cache (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    print(f"  ✓ Model loaded successfully (local)!")

    # --- Prepare model for k-bit training ---
    if device == "cuda":
        model = prepare_model_for_kbit_training(model)
        print("  ✓ Model prepared for quantized training")

    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.1f}M")

    return model, tokenizer


# ============================================================================
# STEP 3: CONFIGURE LoRA ADAPTERS
# ============================================================================

def configure_lora(model):
    """
    Apply LoRA (Low-Rank Adaptation) configuration to the model.
    Only trains ~0.5% of parameters — extremely memory efficient.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Configuring LoRA adapters...")
    print("=" * 60)

    # --- Define LoRA Configuration ---
    lora_config = LoraConfig(
        r=LORA_R,                              # Rank of decomposition
        lora_alpha=LORA_ALPHA,                 # Scaling factor
        target_modules=LORA_TARGET_MODULES,    # Which layers to adapt
        lora_dropout=LORA_DROPOUT,             # Regularization
        bias=LORA_BIAS,                        # No bias training
        task_type=TaskType.CAUSAL_LM,          # Causal language modeling task
    )

    print(f"  LoRA rank (r):         {LORA_R}")
    print(f"  LoRA alpha:            {LORA_ALPHA}")
    print(f"  Scaling factor:        {LORA_ALPHA / LORA_R}")
    print(f"  Target modules:        {LORA_TARGET_MODULES}")
    print(f"  Dropout:               {LORA_DROPOUT}")
    print(f"  Bias:                  {LORA_BIAS}")

    # FIX FOR T4 GPU: Tell PEFT the model is float32 so it doesn't 
    # initialize the new LoRA weights as bfloat16 (which crashes T4).
    if hasattr(model, "config"):
        model.config.torch_dtype = torch.float32

    # --- Apply LoRA to the model ---
    model = get_peft_model(model, lora_config)

    # Cast LoRA adapters and norms to float32 to be absolutely safe.
    # We selectively cast modules to avoid corrupting the 4-bit base weights.
    for name, module in model.named_modules():
        if "lora_" in name or "norm" in name:
            module.to(torch.float32)

    # Print trainable parameter stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = (trainable_params / total_params) * 100

    print(f"\n  ✓ LoRA applied successfully!")
    print(f"  Trainable parameters:  {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"  Frozen parameters:     {total_params - trainable_params:,}")
    print(f"  Total parameters:      {total_params:,}")

    return model, lora_config


# ============================================================================
# STEP 4: LOAD AND PREPARE DATASET (databricks-dolly-15k)
# ============================================================================

def format_alpaca(example: dict) -> str:
    """
    Convert a single Dolly example into Alpaca instruction format.
    Uses the 'instruction', 'context', and 'response' fields.

    Alpaca format:
        ### Instruction:
        {instruction}

        ### Input:
        {context}          (only if context is non-empty)

        ### Response:
        {response}
    """
    instruction = (example.get("instruction") or "").strip()
    context = (example.get("context") or "").strip()
    response = (example.get("response") or "").strip()

    # Build the Alpaca-formatted text
    if context:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    return text


def load_and_prepare_dataset():
    """
    Load databricks/databricks-dolly-15k from HuggingFace Hub.
    - Shuffle with seed=42
    - Select first 2000 samples
    - Convert to Alpaca instruction format ('text' column)
    - Remove empty and duplicate samples
    - Split into 80% train / 20% validation
    - Print detailed dataset statistics
    """
    print("\n" + "=" * 60)
    print("STEP 4: Loading and preparing dataset...")
    print("=" * 60)

    # --- 4a: Download dataset from HuggingFace Hub ---
    print(f"  Downloading '{DATASET_NAME}' from HuggingFace Hub...")
    try:
        raw_dataset = hf_load_dataset(DATASET_NAME, split="train")
        original_size = len(raw_dataset)
        print(f"  ✓ Dataset downloaded — {original_size:,} total samples")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset '{DATASET_NAME}': {e}\n"
            f"Check your internet connection and the dataset name."
        )

    # --- 4b: Shuffle and select subset ---
    print(f"  Shuffling with seed={DATASET_SEED} and selecting {DATASET_SAMPLE_SIZE} samples...")
    raw_dataset = raw_dataset.shuffle(seed=DATASET_SEED)
    raw_dataset = raw_dataset.select(range(min(DATASET_SAMPLE_SIZE, len(raw_dataset))))
    print(f"  ✓ Selected {len(raw_dataset):,} samples from {original_size:,}")

    # --- 4c: Convert to Alpaca instruction format ---
    print("  Converting to Alpaca instruction format...")
    raw_dataset = raw_dataset.map(
        lambda example: {"text": format_alpaca(example)},
        desc="Formatting to Alpaca",
    )

    # --- 4d: Remove empty samples ---
    before_clean = len(raw_dataset)
    raw_dataset = raw_dataset.filter(
        lambda example: len(example["text"].strip()) > 20,
        desc="Removing empty samples",
    )
    empty_removed = before_clean - len(raw_dataset)
    if empty_removed > 0:
        print(f"  ✓ Removed {empty_removed} empty/near-empty samples")

    # --- 4e: Remove duplicate samples ---
    # CRITICAL: We deduplicate the entire dataset BEFORE splitting. 
    # If we split first and deduplicate later, identical samples could end up
    # in BOTH the training and validation sets, causing data leakage and 
    # artificially inflating validation scores (overfitting).
    before_dedup = len(raw_dataset)
    seen_texts = set()
    unique_indices = []
    for i, example in enumerate(raw_dataset):
        text_hash = hashlib.sha256(example["text"].strip().encode("utf-8")).hexdigest()
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            unique_indices.append(i)
    raw_dataset = raw_dataset.select(unique_indices)
    dupes_removed = before_dedup - len(raw_dataset)
    if dupes_removed > 0:
        print(f"  ✓ Removed {dupes_removed} duplicate samples")

    print(f"  ✓ Clean dataset size: {len(raw_dataset):,} samples")

    # --- 4f: Split into train (80%) and validation (20%) ---
    # CRITICAL: Since preprocessing (cleaning, formatting, deduplication) is complete,
    # we can now safely split. The validation data is held out from this point 
    # forward and will never be seen by the model during the training gradient updates.
    print(f"  Splitting into {int((1-VALIDATION_SPLIT)*100)}% train / {int(VALIDATION_SPLIT*100)}% validation...")
    split_dataset = raw_dataset.train_test_split(
        test_size=VALIDATION_SPLIT,
        seed=DATASET_SEED,
        shuffle=True,
    )
    # Rename 'test' to 'validation' for clarity in logs
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    # --- Sanity check: Ensure no identical texts exist across splits ---
    train_texts = set(train_dataset["text"])
    val_texts = set(val_dataset["text"])
    overlap = train_texts.intersection(val_texts)
    assert len(overlap) == 0, f"CRITICAL: Data leakage detected! {len(overlap)} overlapping samples."

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"  ✓ Train samples:      {train_size:,}")
    print(f"  ✓ Validation samples: {val_size:,}")

    # --- 4g: Print detailed dataset statistics ---
    print("\n" + "-" * 60)
    print("  DATASET STATISTICS")
    print("-" * 60)
    print(f"  Original dataset size:   {original_size:,}")
    print(f"  Selected sample size:    {DATASET_SAMPLE_SIZE:,}")
    print(f"  After cleaning:          {len(raw_dataset):,}")
    print(f"  Empty removed:           {empty_removed}")
    print(f"  Duplicates removed:      {dupes_removed}")
    print(f"  Train set size:          {train_size:,}")
    print(f"  Validation set size:     {val_size:,}")

    # Count categories from the original 'category' column
    try:
        categories = raw_dataset["category"]
        category_counts = Counter(categories)
        print(f"\n  Categories ({len(category_counts)}):")
        for cat, count in category_counts.most_common():
            print(f"    {cat:<30} {count:>5} samples")
    except (KeyError, TypeError):
        print("  Categories: (not available in dataset)")

    # Show a few random formatted samples
    print(f"\n  Sample formatted texts:")
    random.seed(DATASET_SEED)
    sample_indices = random.sample(range(train_size), min(3, train_size))
    for idx, i in enumerate(sample_indices):
        text = train_dataset[i]["text"]
        # Show first 120 characters of each sample
        preview = text[:120].replace("\n", " ↵ ")
        if len(text) > 120:
            preview += "..."
        print(f"    [{idx+1}] {preview}")

    print("-" * 60)

    # --- 4h: Keep only the 'text' column for training ---
    # SFTTrainer only needs the 'text' column. Removing extra columns
    # (instruction, context, response, category) prevents Trainer from
    # trying to pass them to the model's forward(), which would crash.
    keep_columns = ["text"]
    remove_cols = [c for c in train_dataset.column_names if c not in keep_columns]
    if remove_cols:
        train_dataset = train_dataset.remove_columns(remove_cols)
        val_dataset = val_dataset.remove_columns(remove_cols)
        print(f"  ✓ Removed extra columns: {remove_cols}")

    return train_dataset, val_dataset


# ============================================================================
# STEP 5: CONFIGURE TRAINING ARGUMENTS
# ============================================================================

def get_training_args(output_dir: str, device: str):
    """
    Set up training hyperparameters optimized for Colab free T4 GPU.
    Now includes validation evaluation during training.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Configuring training arguments...")
    print("=" * 60)

    kwargs = dict(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,     # Eval batch size matches train
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=FP16_ENABLED and device == "cuda",
        bf16=False,
        save_strategy="steps",                     # Explicit — must align with eval
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,                     # Validate every N steps
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_total_limit=2,            # Keep only last 2 checkpoints to save space
        load_best_model_at_end=True,   # Load the best checkpoint at end of training
        metric_for_best_model="eval_loss",  # Use validation loss to pick best model
        greater_is_better=False,       # Lower eval_loss is better
        report_to="none",             # No wandb/tensorboard (keep it simple)
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        gradient_checkpointing=True,   # Saves memory at cost of speed
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Avoids PyTorch 2.x warnings
        remove_unused_columns=False,
        dataloader_pin_memory=True if device == "cuda" else False,
    )

    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    if HAS_SFT_CONFIG:
        config_sig = inspect.signature(SFTConfig.__init__)
        if "max_seq_length" in config_sig.parameters:
            kwargs["max_seq_length"] = MAX_SEQ_LENGTH
        if "dataset_text_field" in config_sig.parameters:
            kwargs["dataset_text_field"] = "text"
        if "packing" in config_sig.parameters:
            kwargs["packing"] = False
        training_args = SFTConfig(**kwargs)
    else:
        training_args = TrainingArguments(**kwargs)

    print(f"  Epochs:                {NUM_TRAIN_EPOCHS}")
    print(f"  Batch size:            {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch size:  {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate:         {LEARNING_RATE}")
    print(f"  FP16:                  {FP16_ENABLED and device == 'cuda'}")
    print(f"  Warmup ratio:          {WARMUP_RATIO}")
    print(f"  Save every:            {SAVE_STEPS} steps")
    print(f"  Log every:             {LOGGING_STEPS} steps")
    print(f"  Eval every:            {EVAL_STEPS} steps")
    print(f"  Optimizer:             {'paged_adamw_8bit' if device == 'cuda' else 'adamw_torch'}")
    print(f"  Best model selection:  eval_loss (lower is better)")

    return training_args


# ============================================================================
# STEP 6: TRAIN WITH SFTTrainer
# ============================================================================

def train_model(model, tokenizer, train_dataset, val_dataset, training_args, lora_config):
    """
    Fine-tune the model using SFTTrainer from the TRL library.
    SFTTrainer handles tokenization and formatting automatically.
    Now includes validation dataset for eval during training.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Starting LoRA fine-tuning with SFTTrainer...")
    print("=" * 60)
    print("  This will take ~30-60 minutes on a T4 GPU.")
    print("  Training & validation loss will be printed periodically.\n")

    try:
        # --- Initialize SFTTrainer ---
        import inspect
        
        # Add EarlyStoppingCallback to halt training if validation loss stops improving.
        # This explicitly prevents overfitting to the training set.
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        
        trainer_kwargs = dict(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            callbacks=callbacks,
        )

        # Handle API changes across TRL/Transformers versions gracefully
        sig = inspect.signature(SFTTrainer.__init__)
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        # Only pass these to SFTTrainer if they weren't already accepted by SFTConfig
        config_sig_params = []
        if HAS_SFT_CONFIG:
            config_sig_params = inspect.signature(SFTConfig.__init__).parameters

        if "max_seq_length" in sig.parameters and "max_seq_length" not in config_sig_params:
            trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
        if "dataset_text_field" in sig.parameters and "dataset_text_field" not in config_sig_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "packing" in sig.parameters and "packing" not in config_sig_params:
            trainer_kwargs["packing"] = False

        trainer = SFTTrainer(**trainer_kwargs)

        print("  ✓ SFTTrainer initialized")
        print(f"  ✓ Train samples:      {len(train_dataset):,}")
        print(f"  ✓ Validation samples: {len(val_dataset):,}")
        print("  Starting training...\n")
        print("-" * 60)

        # --- Run Training ---
        train_result = trainer.train()

        print("-" * 60)
        print(f"\n  ✓ Training completed!")
        print(f"  Total training loss:    {train_result.training_loss:.4f}")
        print(f"  Training steps:         {train_result.global_step}")

        # --- Run final validation and compute perplexity ---
        # Wrapped in try/except so a failed eval doesn't lose the trained model
        eval_results = {}
        try:
            print("\n  Running final validation...")
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get("eval_loss", float("nan"))
            eval_perplexity = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")
        except Exception as eval_err:
            print(f"  ⚠ Evaluation failed: {eval_err}")
            print("  Training was successful — skipping final eval metrics.")
            eval_loss = float("nan")
            eval_perplexity = float("nan")

        print(f"\n  ╔══════════════════════════════════════════════╗")
        print(f"  ║          TRAINING RESULTS SUMMARY            ║")
        print(f"  ╠══════════════════════════════════════════════╣")
        print(f"  ║  Training Loss:     {train_result.training_loss:>10.4f}             ║")
        print(f"  ║  Validation Loss:   {eval_loss:>10.4f}             ║")
        print(f"  ║  Perplexity:        {eval_perplexity:>10.2f}             ║")
        print(f"  ║  Total Steps:       {train_result.global_step:>10d}             ║")
        print(f"  ╚══════════════════════════════════════════════╝")

        return trainer, eval_results

    except Exception as e:
        print(f"  ✗ Training error: {e}")
        raise


# ============================================================================
# STEP 7: SAVE LoRA ADAPTER WEIGHTS
# ============================================================================

def save_adapter(trainer, output_dir: str, tokenizer):
    """
    Save the trained LoRA adapter weights and tokenizer.
    Only the adapter is saved (~10-20 MB), not the full model (~4GB).
    """
    print("\n" + "=" * 60)
    print("STEP 7: Saving LoRA adapter weights...")
    print("=" * 60)

    try:
        # Save the adapter
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Calculate and print saved file sizes
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                fpath = os.path.join(root, f)
                fsize = os.path.getsize(fpath)
                total_size += fsize
                file_count += 1

        print(f"  ✓ Adapter saved to: {output_dir}")
        print(f"  Files saved:        {file_count}")
        print(f"  Total size:         {total_size / 1e6:.1f} MB")

        # List saved files
        print(f"\n  Saved files:")
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                fsize = os.path.getsize(fpath)
                print(f"    {f} ({fsize / 1024:.1f} KB)")

    except Exception as e:
        print(f"  ✗ Error saving adapter: {e}")
        raise


# ============================================================================
# STEP 8: SAVE TRAINING HISTORY
# ============================================================================

def save_training_history(trainer, output_dir: str):
    """
    Export training history (loss, validation loss, learning rate) to a CSV.
    Useful for plotting and for the final report.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Saving training history...")
    print("=" * 60)

    try:
        import csv
        history_file = os.path.join(output_dir, "training_history.csv")
        
        log_history = trainer.state.log_history
        steps_data = {}
        
        for entry in log_history:
            step = entry.get("step")
            if step is None:
                continue
            
            if step not in steps_data:
                steps_data[step] = {"step": step, "epoch": entry.get("epoch", "")}
                
            if "loss" in entry:
                steps_data[step]["loss"] = entry["loss"]
            if "eval_loss" in entry:
                steps_data[step]["eval_loss"] = entry["eval_loss"]
            if "learning_rate" in entry:
                steps_data[step]["learning_rate"] = entry["learning_rate"]
                
        with open(history_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "epoch", "learning_rate", "loss", "eval_loss"])
            
            for step in sorted(steps_data.keys()):
                data = steps_data[step]
                writer.writerow([
                    data.get("step", ""),
                    f"{data.get('epoch'):.4f}" if isinstance(data.get("epoch"), (float, int)) else data.get("epoch", ""),
                    f"{data.get('learning_rate'):.2e}" if isinstance(data.get("learning_rate"), (float, int)) else data.get("learning_rate", ""),
                    f"{data.get('loss'):.4f}" if isinstance(data.get("loss"), (float, int)) else data.get("loss", ""),
                    f"{data.get('eval_loss'):.4f}" if isinstance(data.get("eval_loss"), (float, int)) else data.get("eval_loss", "")
                ])
                
        print(f"  ✓ Training history saved to: {history_file}")
    except Exception as e:
        print(f"  ✗ Error saving training history: {e}")


# ============================================================================
# STEP 9: GENERATE TRAINING PLOTS
# ============================================================================

def plot_training_history(history_file: str, output_dir: str):
    """
    Generate and save a loss curve plot for the final report.
    """
    print("\n" + "=" * 60)
    print("STEP 9: Generating training plots...")
    print("=" * 60)
    
    try:
        import csv
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        steps = []
        train_loss = []
        val_steps = []
        val_loss = []
        
        with open(history_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row['step'])
                if row['loss']:
                    steps.append(step)
                    train_loss.append(float(row['loss']))
                if row['eval_loss']:
                    val_steps.append(step)
                    val_loss.append(float(row['eval_loss']))
                    
        plt.figure(figsize=(10, 6))
        
        if steps and train_loss:
            plt.plot(steps, train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
        if val_steps and val_loss:
            plt.plot(val_steps, val_loss, label='Validation Loss', color='#d62728', marker='o', linewidth=2)
            
        plt.title('Training and Validation Loss', fontsize=14, pad=15)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"  ✓ Training plot saved to: {plot_path}")
    except Exception as e:
        print(f"  ✗ Error generating plot: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main training pipeline:
    GPU Check → Load Model → LoRA Config → Load Data → Train → Save
    """
    print("\n" + "★" * 60)
    print("  MyStyle Writer — LoRA Fine-Tuning Pipeline")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Dataset: {DATASET_NAME} ({DATASET_SAMPLE_SIZE} samples)")
    print("  Method:  LoRA (Low-Rank Adaptation) + QLoRA 4-bit")
    print("★" * 60 + "\n")

    try:
        # Step 0: Validate local model
        if not validate_local_model(MODEL_NAME):
            print("\n❌ Cannot proceed without the local model. Exiting.")
            sys.exit(1)

        # Step 1: Check GPU
        device = check_gpu()

        # Step 2: Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)

        # Step 3: Configure LoRA
        model, lora_config = configure_lora(model)

        # Step 4: Load and prepare dataset (Dolly-15k → Alpaca format)
        train_dataset, val_dataset = load_and_prepare_dataset()

        # Step 5: Configure training arguments (with validation)
        training_args = get_training_args(ADAPTER_OUTPUT_DIR, device)

        # Step 6: Train the model (with validation evaluation)
        trainer, eval_results = train_model(
            model, tokenizer, train_dataset, val_dataset, training_args, lora_config
        )

        # Step 7: Save the adapter
        save_adapter(trainer, ADAPTER_OUTPUT_DIR, tokenizer)

        # Step 8: Save training history
        save_training_history(trainer, ADAPTER_OUTPUT_DIR)

        # Step 9: Generate training plots
        history_file = os.path.join(ADAPTER_OUTPUT_DIR, "training_history.csv")
        plot_training_history(history_file, ADAPTER_OUTPUT_DIR)

        # --- Final Summary ---
        eval_loss = eval_results.get("eval_loss", float("nan"))
        eval_ppl = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")

        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"  Adapter saved at:   {ADAPTER_OUTPUT_DIR}")
        print(f"  Base model:         {MODEL_NAME}")
        print(f"  Dataset:            {DATASET_NAME}")
        print(f"  Training epochs:    {NUM_TRAIN_EPOCHS}")
        print(f"  LoRA rank:          {LORA_R}")
        print(f"  Final val loss:     {eval_loss:.4f}")
        print(f"  Final perplexity:   {eval_ppl:.2f}")
        print(f"\n  Next steps:")
        print(f"    1. Run evaluate.py to compare base vs fine-tuned model")
        print(f"    2. Run app.py to launch the Gradio web UI")
        print("=" * 60 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ FILE ERROR: {e}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ GPU OUT OF MEMORY!")
            print("   Try reducing BATCH_SIZE to 1 or MAX_SEQ_LENGTH to 256.")
            print("   Make sure you're using a T4 GPU on Colab.")
        else:
            print(f"\n❌ RUNTIME ERROR: {e}")
        raise
    except KeyboardInterrupt:
        print(f"\n⚠ TRAINING INTERRUPTED by user.")
        print("   Partial checkpoints may have been saved in the checkpoints/ folder.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
