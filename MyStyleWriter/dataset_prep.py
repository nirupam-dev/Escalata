"""
==============================================================================
SCRIPT 1 — Dataset Preparation (dataset_prep.py)
Project #27: MyStyle Writer — Style Fine-Tuning
==============================================================================

This script:
  1. Loads raw .txt file from ./data/raw_text.txt
  2. Cleans text: removes special characters, fixes encoding, removes short lines
  3. Converts cleaned text into Alpaca instruction format (matching train.py):
        ### Instruction:
        Continue writing in this style.

        ### Input:
        {first 50 words of sample}

        ### Response:
        {remaining text of sample}
  4. Deduplicates using SHA256 hashing (same method as train.py)
  5. Splits 80% train / 20% validation with seed=42
  6. Saves as HuggingFace Arrow format to ./data/prepared/
  7. Prints full statistics

Usage (run locally or on Colab):
    python dataset_prep.py

Prerequisites:
    - Place your raw text file at ./data/raw_text.txt
    - pip install datasets
==============================================================================
"""

import os
import re
import sys
import hashlib
import unicodedata
from datasets import Dataset, DatasetDict

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Path Configuration ---
# Resolve script directory (works in both Colab and local environments)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()  # Colab fallback → /content

# Path to the raw text file provided by the user
RAW_TEXT_FILE = os.path.join(SCRIPT_DIR, "data", "raw_text.txt")

# Where to save the processed HuggingFace dataset (Arrow format)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "prepared")

# --- Cleaning Configuration ---
# Minimum number of characters a line must have to be kept
MIN_LINE_LENGTH = 20

# --- Split Configuration ---
TRAIN_RATIO = 0.80          # 80% train
VALIDATION_RATIO = 0.20     # 20% validation
RANDOM_SEED = 42            # Reproducible seed (matches train.py)

# --- Alpaca Format Configuration ---
# Number of words from each sample to use as the "Input" section
INPUT_WORD_COUNT = 50

# The instruction text used for all samples
INSTRUCTION_TEXT = "Continue writing in this style."


# ============================================================================
# STEP 1: LOAD RAW TEXT FILE
# ============================================================================

def load_raw_text(file_path: str) -> list:
    """
    Load the raw text file from ./data/raw_text.txt.
    Reads the entire file and splits into individual paragraphs.
    
    Returns:
        list: A list of paragraph strings
    """
    print("=" * 60)
    print("STEP 1: Loading raw text file...")
    print("=" * 60)

    # --- Check if the file exists ---
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Raw text file not found at: {file_path}\n"
            f"Please create ./data/raw_text.txt with your writing samples."
        )

    # --- Read the file with encoding fallback ---
    try:
        # Try UTF-8 first (most common encoding)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"  ✓ Read file with UTF-8 encoding")
        except UnicodeDecodeError:
            # Fall back to latin-1 if UTF-8 fails
            print(f"  ⚠ UTF-8 failed, trying latin-1 encoding...")
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
            print(f"  ✓ Read file with latin-1 encoding")

    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")

    # --- Split content into paragraphs ---
    # Split on double newlines (paragraph breaks) or single newlines
    # First try double newline split for paragraph-based text
    paragraphs = re.split(r'\n\s*\n', content)
    
    # If we got very few paragraphs, try single newline split
    if len(paragraphs) < 5:
        paragraphs = content.strip().split("\n")

    # Remove empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # --- Print loading statistics ---
    file_size = os.path.getsize(file_path)
    print(f"  File: {os.path.basename(file_path)}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Total characters: {len(content):,}")
    print(f"  Total paragraphs found: {len(paragraphs)}")

    if not paragraphs:
        raise ValueError(
            f"No text found in {file_path}. "
            f"The file appears to be empty."
        )

    return paragraphs


# ============================================================================
# STEP 2: CLEAN TEXT
# ============================================================================

def clean_text(paragraphs: list, min_length: int) -> list:
    """
    Clean the raw text paragraphs:
      - Normalize Unicode characters (fix encoding artifacts like smart quotes)
      - Remove special characters (keep letters, numbers, basic punctuation)
      - Collapse multiple spaces into one
      - Remove lines shorter than min_length characters
      - Strip leading/trailing whitespace
    
    Args:
        paragraphs: List of raw paragraph strings
        min_length: Minimum character count to keep a paragraph
    
    Returns:
        list: Cleaned paragraphs that passed all filters
    """
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning text data...")
    print("=" * 60)

    cleaned_passages = []
    total_raw = len(paragraphs)
    removed_empty = 0
    removed_short = 0
    removed_special = 0

    for paragraph in paragraphs:
        # --- Normalize Unicode ---
        # Converts fancy quotes ("), em-dashes (—), etc. to ASCII equivalents
        try:
            paragraph = unicodedata.normalize("NFKD", paragraph)
        except Exception:
            pass  # Skip normalization if it fails

        # --- Strip whitespace ---
        paragraph = paragraph.strip()

        # --- Skip empty paragraphs ---
        if not paragraph:
            removed_empty += 1
            continue

        # --- Remove special characters ---
        # Keep: letters, digits, spaces, and basic punctuation (.,!?;:'-")
        paragraph = re.sub(r"[^\w\s.,!?;:'\"-]", "", paragraph)

        # --- Collapse multiple spaces/newlines into single space ---
        paragraph = re.sub(r"\s+", " ", paragraph).strip()

        # --- Check if paragraph is now empty after cleaning ---
        if not paragraph:
            removed_special += 1
            continue

        # --- Drop paragraphs shorter than minimum length ---
        if len(paragraph) < min_length:
            removed_short += 1
            continue

        # --- Paragraph passed all filters ---
        cleaned_passages.append(paragraph)

    # --- Print cleaning statistics ---
    print(f"  Total raw paragraphs:    {total_raw}")
    print(f"  Removed (empty):         {removed_empty}")
    print(f"  Removed (too short):     {removed_short}")
    print(f"  Removed (special chars): {removed_special}")
    print(f"  Paragraphs kept:         {len(cleaned_passages)}")
    print(f"  Min length filter:       {min_length} characters")

    if not cleaned_passages:
        raise ValueError(
            "No paragraphs survived cleaning! "
            "Try lowering MIN_LINE_LENGTH or adding more text to raw_text.txt."
        )

    # --- Show sample cleaned passages ---
    print(f"\n  Sample cleaned passages:")
    for i, passage in enumerate(cleaned_passages[:3]):
        preview = passage[:100] + "..." if len(passage) > 100 else passage
        print(f"    [{i+1}] {preview}")

    return cleaned_passages


# ============================================================================
# STEP 3: CONVERT TO ALPACA INSTRUCTION FORMAT
# ============================================================================

def convert_to_alpaca_format(passages: list, input_word_count: int) -> list:
    """
    Convert cleaned text passages into Alpaca instruction format.
    This EXACTLY matches how train.py expects the data:
    
        ### Instruction:
        Continue writing in this style.

        ### Input:
        {first 50 words of sample}

        ### Response:
        {remaining text of sample}
    
    Args:
        passages: List of cleaned text strings
        input_word_count: Number of words to use as the "Input" section
    
    Returns:
        list: List of Alpaca-formatted text strings
    """
    print("\n" + "=" * 60)
    print("STEP 3: Converting to Alpaca instruction format...")
    print("=" * 60)

    formatted_samples = []
    skipped_too_short = 0

    for passage in passages:
        # --- Split passage into words ---
        words = passage.split()

        # --- Skip passages that are too short to split meaningfully ---
        # We need at least input_word_count + 10 words to have a useful Response
        if len(words) < input_word_count + 10:
            skipped_too_short += 1
            continue

        # --- Split into Input (first N words) and Response (remaining words) ---
        input_text = " ".join(words[:input_word_count])
        response_text = " ".join(words[input_word_count:])

        # --- Build the Alpaca-formatted string ---
        alpaca_text = (
            f"### Instruction:\n"
            f"{INSTRUCTION_TEXT}\n\n"
            f"### Input:\n"
            f"{input_text}\n\n"
            f"### Response:\n"
            f"{response_text}"
        )

        formatted_samples.append(alpaca_text)

    # --- Print conversion statistics ---
    print(f"  Total cleaned passages:     {len(passages)}")
    print(f"  Skipped (too short):        {skipped_too_short}")
    print(f"  Successfully formatted:     {len(formatted_samples)}")
    print(f"  Input section word count:   {input_word_count} words")

    if not formatted_samples:
        raise ValueError(
            "No samples could be converted to Alpaca format! "
            "Your passages may be too short. Each passage needs at least "
            f"{input_word_count + 10} words."
        )

    # --- Show a sample formatted text ---
    print(f"\n  Sample Alpaca-formatted text:")
    sample = formatted_samples[0]
    preview_lines = sample.split("\n")[:8]
    for line in preview_lines:
        truncated = line[:80] + "..." if len(line) > 80 else line
        print(f"    {truncated}")
    if len(sample.split("\n")) > 8:
        print(f"    ...")

    return formatted_samples


# ============================================================================
# STEP 4: DEDUPLICATE USING SHA256 HASHING
# ============================================================================

def deduplicate_samples(samples: list) -> list:
    """
    Remove duplicate samples using SHA256 hashing.
    This uses the SAME deduplication method as train.py to ensure consistency.
    Each sample's text is hashed, and only unique hashes are kept.
    
    Args:
        samples: List of Alpaca-formatted text strings
    
    Returns:
        list: Deduplicated list of samples
    """
    print("\n" + "=" * 60)
    print("STEP 4: Deduplicating with SHA256 hashing...")
    print("=" * 60)

    seen_hashes = set()
    unique_samples = []
    duplicates_found = 0

    for sample in samples:
        # --- Compute SHA256 hash of the sample text ---
        # .strip() and .encode("utf-8") match exactly how train.py hashes
        text_hash = hashlib.sha256(sample.strip().encode("utf-8")).hexdigest()

        # --- Check if this hash has been seen before ---
        if text_hash in seen_hashes:
            duplicates_found += 1
            continue

        # --- New unique sample ---
        seen_hashes.add(text_hash)
        unique_samples.append(sample)

    # --- Print deduplication statistics ---
    print(f"  Total samples before:    {len(samples)}")
    print(f"  Duplicates removed:      {duplicates_found}")
    print(f"  Unique samples after:    {len(unique_samples)}")

    if not unique_samples:
        raise ValueError("All samples were duplicates! Add more diverse text.")

    return unique_samples


# ============================================================================
# STEP 5: SPLIT INTO TRAIN / VALIDATION SETS
# ============================================================================

def split_dataset(samples: list, train_ratio: float, seed: int) -> DatasetDict:
    """
    Split the deduplicated samples into train (80%) and validation (20%) sets.
    Uses HuggingFace's train_test_split with seed=42 for reproducibility.
    
    IMPORTANT: Deduplication is done BEFORE splitting (Step 4) to prevent
    data leakage — identical samples cannot appear in both splits.
    
    Args:
        samples: List of unique Alpaca-formatted text strings
        train_ratio: Fraction for training (0.80)
        seed: Random seed (42)
    
    Returns:
        DatasetDict: HuggingFace DatasetDict with 'train' and 'validation' splits
    """
    print("\n" + "=" * 60)
    print("STEP 5: Splitting into train/validation sets...")
    print("=" * 60)

    # --- Create HuggingFace Dataset from the list ---
    full_dataset = Dataset.from_dict({"text": samples})
    print(f"  Full dataset size: {len(full_dataset)} samples")

    # --- Split into train and validation ---
    validation_ratio = 1.0 - train_ratio
    split_dataset = full_dataset.train_test_split(
        test_size=validation_ratio,
        seed=seed,
        shuffle=True
    )

    # --- Rename 'test' to 'validation' for clarity ---
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

    train_size = len(dataset_dict["train"])
    val_size = len(dataset_dict["validation"])

    # --- Print split statistics ---
    print(f"  Train set:      {train_size} samples ({train_ratio*100:.0f}%)")
    print(f"  Validation set: {val_size} samples ({validation_ratio*100:.0f}%)")
    print(f"  Random seed:    {seed}")

    # --- Verify no data leakage ---
    try:
        train_texts = set(dataset_dict["train"]["text"])
        val_texts = set(dataset_dict["validation"]["text"])
        overlap = train_texts.intersection(val_texts)
        if len(overlap) > 0:
            print(f"  ⚠ WARNING: {len(overlap)} overlapping samples detected!")
        else:
            print(f"  ✓ No data leakage — zero overlap between splits")
    except Exception as e:
        print(f"  ⚠ Could not verify leakage: {e}")

    return dataset_dict


# ============================================================================
# STEP 6: SAVE DATASET IN HUGGINGFACE ARROW FORMAT
# ============================================================================

def save_dataset(dataset_dict: DatasetDict, output_dir: str) -> None:
    """
    Save the DatasetDict to disk in HuggingFace Arrow format.
    Arrow format enables fast loading during training with zero deserialization.
    
    Args:
        dataset_dict: HuggingFace DatasetDict with train/validation splits
        output_dir: Path to save the dataset (./data/prepared/)
    """
    print("\n" + "=" * 60)
    print("STEP 6: Saving dataset in Arrow format...")
    print("=" * 60)

    try:
        # --- Create output directory if it doesn't exist ---
        os.makedirs(output_dir, exist_ok=True)

        # --- Save the DatasetDict ---
        dataset_dict.save_to_disk(output_dir)
        print(f"  ✓ Dataset saved to: {output_dir}")

        # --- Print saved file sizes ---
        for split_name in ["train", "validation"]:
            split_dir = os.path.join(output_dir, split_name)
            if os.path.exists(split_dir):
                total_size = sum(
                    os.path.getsize(os.path.join(split_dir, f))
                    for f in os.listdir(split_dir)
                    if os.path.isfile(os.path.join(split_dir, f))
                )
                print(f"    {split_name}/ — {total_size / 1024:.1f} KB")

    except Exception as e:
        print(f"  ✗ Error saving dataset: {e}")
        raise


# ============================================================================
# STEP 7: PRINT FINAL STATISTICS
# ============================================================================

def print_final_statistics(dataset_dict: DatasetDict, total_raw: int,
                           removed_empty: int, removed_duplicates: int) -> None:
    """
    Print comprehensive statistics about the entire preparation pipeline.
    
    Args:
        dataset_dict: Final HuggingFace DatasetDict
        total_raw: Total raw paragraphs loaded from file
        removed_empty: Number of empty/short paragraphs removed
        removed_duplicates: Number of duplicate samples removed
    """
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)

    # --- Overall pipeline summary ---
    train_size = len(dataset_dict["train"])
    val_size = len(dataset_dict["validation"])
    total_final = train_size + val_size

    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║         PREPARATION PIPELINE SUMMARY         ║")
    print(f"  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Raw paragraphs loaded:    {total_raw:>10}         ║")
    print(f"  ║  Removed (empty/short):    {removed_empty:>10}         ║")
    print(f"  ║  Removed (duplicates):     {removed_duplicates:>10}         ║")
    print(f"  ║  Final samples:            {total_final:>10}         ║")
    print(f"  ║  Train set size:           {train_size:>10}         ║")
    print(f"  ║  Validation set size:      {val_size:>10}         ║")
    print(f"  ╚══════════════════════════════════════════════╝")

    # --- Per-split detailed statistics ---
    for split_name in ["train", "validation"]:
        split = dataset_dict[split_name]
        texts = split["text"]

        # Calculate text statistics
        total_chars = sum(len(t) for t in texts)
        total_words = sum(len(t.split()) for t in texts)
        avg_chars = total_chars / len(texts) if texts else 0
        avg_words = total_words / len(texts) if texts else 0

        # Calculate vocabulary diversity
        all_words = []
        for t in texts:
            all_words.extend(t.lower().split())
        unique_words = len(set(all_words))
        vocab_diversity = unique_words / len(all_words) if all_words else 0

        print(f"\n  [{split_name.upper()}]")
        print(f"    Samples:              {len(texts)}")
        print(f"    Total characters:     {total_chars:,}")
        print(f"    Total words:          {total_words:,}")
        print(f"    Avg chars/sample:     {avg_chars:.1f}")
        print(f"    Avg words/sample:     {avg_words:.1f}")
        print(f"    Unique vocabulary:    {unique_words:,} words")
        print(f"    Vocabulary diversity: {vocab_diversity:.2%}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main pipeline: Load → Clean → Format → Deduplicate → Split → Save → Stats
    
    This prepares user-provided text for fine-tuning with train.py.
    The output matches the Alpaca instruction format that train.py expects.
    """
    print("\n" + "★" * 60)
    print("  MyStyle Writer — Dataset Preparation Pipeline")
    print("  Converts raw text → Alpaca instruction format")
    print("  Output: HuggingFace Arrow dataset for train.py")
    print("★" * 60 + "\n")

    try:
        # Step 1: Load raw text from ./data/raw_text.txt
        paragraphs = load_raw_text(RAW_TEXT_FILE)
        total_raw = len(paragraphs)

        # Step 2: Clean text (remove special chars, encoding issues, short lines)
        cleaned_passages = clean_text(paragraphs, MIN_LINE_LENGTH)
        removed_empty = total_raw - len(cleaned_passages)

        # Step 3: Convert to Alpaca instruction format (matching train.py)
        formatted_samples = convert_to_alpaca_format(cleaned_passages, INPUT_WORD_COUNT)

        # Step 4: Deduplicate using SHA256 (same method as train.py)
        unique_samples = deduplicate_samples(formatted_samples)
        removed_duplicates = len(formatted_samples) - len(unique_samples)

        # Step 5: Split into 80% train / 20% validation with seed=42
        dataset_dict = split_dataset(unique_samples, TRAIN_RATIO, RANDOM_SEED)

        # Step 6: Save as HuggingFace Arrow format to ./data/prepared/
        save_dataset(dataset_dict, OUTPUT_DIR)

        # Step 7: Print comprehensive statistics
        print_final_statistics(dataset_dict, total_raw, removed_empty, removed_duplicates)

        # --- Final success message ---
        print("\n" + "=" * 60)
        print("✅ DATASET PREPARATION COMPLETE!")
        print(f"   Saved to: {OUTPUT_DIR}")
        print(f"   Train samples:      {len(dataset_dict['train'])}")
        print(f"   Validation samples: {len(dataset_dict['validation'])}")
        print("\n   Next steps:")
        print("     1. Run train.py to start LoRA fine-tuning")
        print("     2. Run evaluate.py to compare models")
        print("     3. Run app.py to launch the Gradio UI")
        print("=" * 60 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ FILE ERROR: {e}")
        print("   Make sure you have ./data/raw_text.txt with your text samples.")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ DATA ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
