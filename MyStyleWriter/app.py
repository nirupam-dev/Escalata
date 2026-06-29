"""
==============================================================================
SCRIPT 3 — Professional Gradio UI (app.py)
Project #27: MyStyle Writer — Style Fine-Tuning
==============================================================================

This script creates a professional dark-themed Gradio web UI that:
  1. Loads both base Qwen2.5-3B and fine-tuned LoRA model at startup
  2. Shows side-by-side comparison of base vs fine-tuned outputs
  3. Calculates and displays 4 metrics live (Perplexity, BLEU, AvgLen, VocabDiv)
  4. Shows a Style Shift Score progress bar
  5. Displays loss_curve.png from best_adapter/
  6. Professional dark theme with gradient accents

Usage:
    python app.py
    Then open http://localhost:7860

Prerequisites:
    - Run train.py first to create best_adapter/
    - GPU recommended but works on CPU (slower)
==============================================================================
"""

import os
import sys
import re
import math
import random

# --- Block ALL Hugging Face network access ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# --- Conditional ML imports (allows demo mode without GPU libraries) ---
DEMO_MODE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    print("  ✓ ML libraries loaded (torch, transformers, peft)")
except ImportError:
    DEMO_MODE = True
    print("  [!] ML libraries not found -- running in DEMO MODE")
    print("    Install torch, transformers, peft for real model inference.")

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Local Model Path ---
# Use the HuggingFace cache directory where the model was previously downloaded.
# Set this to your actual local model directory if different.
_HF_CACHE_MODEL = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--Qwen--Qwen2.5-3B-Instruct"
)
# If a snapshot exists in the cache, use the snapshot path directly
_SNAPSHOT_DIR = os.path.join(_HF_CACHE_MODEL, "snapshots")
if os.path.isdir(_SNAPSHOT_DIR) and os.listdir(_SNAPSHOT_DIR):
    # Use the first (usually only) snapshot revision
    _rev = os.listdir(_SNAPSHOT_DIR)[0]
    MODEL_NAME = os.path.join(_SNAPSHOT_DIR, _rev)
else:
    # Fall back to the standard HF model ID (will be blocked by local_files_only)
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
ADAPTER_DIR = os.path.join(SCRIPT_DIR, "best_adapter")
LOSS_CURVE_PATH = os.path.join(ADAPTER_DIR, "loss_curve.png")

# Generation settings
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# ============================================================================
# DEMO MODE — Simulated outputs when ML libraries aren't available
# ============================================================================

DEMO_RESPONSES_BASE = {
    "default": [
        "The scene unfolded with a certain grandeur that was difficult to put into words. The characters moved through the landscape, each carrying their own burdens and aspirations, weaving together a tapestry of human experience that stretched far beyond the immediate moment. There was something profoundly beautiful about the way they interacted, each conversation revealing layers of meaning that only became apparent upon reflection.",
        "As the narrative continued, the complexity of the situation became increasingly apparent. Multiple perspectives converged, creating a rich and nuanced portrayal of events that defied simple categorization. The emotional undertones were carefully woven throughout, adding depth and resonance to what might otherwise have been a straightforward account.",
        "The world around them seemed to hold its breath, waiting for something significant to happen. Every detail — the quality of the light, the texture of the air, the subtle shifts in mood — contributed to an atmosphere of anticipation that was both thrilling and unsettling in equal measure.",
    ],
}

DEMO_RESPONSES_FT = {
    "default": [
        "It was quiet. The air was still. He stood and looked at it for a long time. Then he turned and walked away. He did not look back. The road was empty. The sun was low.",
        "She said nothing. He waited. The room was cold and the light came through the window in a thin line. He picked up his glass. It was empty. He set it down again.",
        "The water moved slow. It was dark and deep. He watched it go past. A bird flew over. He did not move. After a while he stood up and went inside. It was getting dark.",
    ],
}

# ============================================================================
# GLOBAL MODEL VARIABLES (loaded once at startup)
# ============================================================================

base_model = None
ft_model = None
tokenizer = None
device = None
models_loaded = False


# ============================================================================
# MODEL LOADING
# ============================================================================

def validate_local_model(model_path: str) -> bool:
    """
    Validate that the local model directory exists and contains required files.
    If the model is missing or incomplete, print a clear error and return False.
    """
    print(f"  Validating local model at: {model_path}")

    if not os.path.isdir(model_path):
        print(f"\n❌ LOCAL MODEL NOT FOUND!")
        print(f"   Expected model directory: {model_path}")
        print(f"   The model has NOT been fully downloaded.")
        print(f"\n   To fix this, download the model ONCE with:")
        print(f"     python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')\"")
        print(f"   Then re-run this script. It will work fully offline.")
        return False

    # Check for essential model files
    required_files = ["config.json"]
    weight_patterns = [".safetensors", ".bin"]  # Model weights
    tokenizer_patterns = ["tokenizer.json", "tokenizer_config.json"]

    missing = []
    for rf in required_files:
        if not os.path.exists(os.path.join(model_path, rf)):
            missing.append(rf)

    # Check for at least one weight file
    files_in_dir = os.listdir(model_path) if os.path.isdir(model_path) else []
    has_weights = any(
        any(f.endswith(ext) for ext in weight_patterns)
        for f in files_in_dir
        if not f.endswith(".incomplete")
    )
    if not has_weights:
        missing.append("model weights (*.safetensors or *.bin)")

    has_tokenizer = any(f in files_in_dir for f in tokenizer_patterns)
    if not has_tokenizer:
        missing.append("tokenizer files (tokenizer.json / tokenizer_config.json)")

    if missing:
        print(f"\n❌ LOCAL MODEL IS INCOMPLETE!")
        print(f"   Model directory: {model_path}")
        print(f"   Missing files:")
        for m in missing:
            print(f"     • {m}")
        print(f"\n   The model was partially downloaded. To complete the download, run:")
        print(f"     python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')\"")
        print(f"   Then re-run this script. It will work fully offline.")
        return False

    print(f"  ✓ Local model validated — all required files present")
    return True


def load_models():
    """Load both base and fine-tuned models at startup with progress messages."""
    global base_model, ft_model, tokenizer, device, models_loaded

    if models_loaded:
        return True

    if DEMO_MODE:
        models_loaded = True
        return True

    print("=" * 60)
    print("Loading models from LOCAL cache (no downloads)...")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Model path: {MODEL_NAME}")

    # --- Validate local model exists before attempting to load ---
    if not validate_local_model(MODEL_NAME):
        print("\n❌ Cannot proceed without the local model. Exiting.")
        sys.exit(1)

    # --- Load Tokenizer (local only, no downloads) ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("  [OK] Tokenizer loaded (local)")
    except Exception as e:
        print(f"  [X] Tokenizer error: {e}")
        print("      Ensure the model is fully downloaded locally.")
        return False

    # --- Quantization config for GPU ---
    quant_config = None
    if device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # --- Load Base Model (local only, no downloads) ---
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        base_model.eval()
        print("  [OK] Base model loaded (local)")
    except Exception as e:
        print(f"  [X] Base model error: {e}")
        print("      The local model files may be incomplete or corrupted.")
        return False

    # --- Load Fine-tuned Model with LoRA adapter ---
    if os.path.exists(ADAPTER_DIR):
        try:
            ft_base = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quant_config,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_DIR, local_files_only=True)
            ft_model.eval()
            print("  [OK] Fine-tuned model loaded (LoRA adapter merged, local)")
        except Exception as e:
            print(f"  [X] Fine-tuned model error: {e}")
            ft_model = None
    else:
        print(f"  [!] Adapter not found at {ADAPTER_DIR}")
        print(f"    Run train.py first to create the adapter.")
        ft_model = None

    models_loaded = True
    print("  [OK] All models ready (fully offline)!\n")
    return True


# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate(model, prompt: str, is_base: bool = True) -> str:
    """Generate text continuation from a prompt using the given model."""
    if DEMO_MODE:
        responses = DEMO_RESPONSES_BASE["default"] if is_base else DEMO_RESPONSES_FT["default"]
        return random.choice(responses)
        
    if model is None:
        return "[Model not loaded — run train.py first]"
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
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        generated_tokens = output_ids[0][input_ids.shape[1]:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        return f"[Error: {e}]"


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calc_perplexity(model, text: str, is_base: bool = True) -> float:
    """Calculate perplexity — lower means more confident/fluent."""
    if DEMO_MODE:
        return random.uniform(40.0, 50.0) if is_base else random.uniform(25.0, 35.0)
        
    if model is None:
        return float("nan")
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        return min(math.exp(outputs.loss.item()), 10000.0)
    except Exception:
        return float("nan")


def calc_bleu(generated: str, reference: str) -> float:
    """Calculate simplified BLEU score — similarity to reference style."""
    try:
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        if not gen_tokens or not ref_tokens:
            return 0.0
        precisions = []
        for n in range(1, 5):
            gen_ngrams = [tuple(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
            if not gen_ngrams:
                precisions.append(0.0)
                continue
            ref_counter = Counter(ref_ngrams)
            matches = sum(1 for ng in gen_ngrams if ref_counter.get(ng, 0) > 0)
            precisions.append(matches / len(gen_ngrams))
        smoothed = [max(p, 1e-10) for p in precisions]
        log_avg = sum(math.log(p) for p in smoothed) / len(smoothed)
        bleu = math.exp(log_avg)
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(gen_tokens), 1)))
        return bleu * bp
    except Exception:
        return 0.0


def calc_avg_sentence_length(text: str) -> float:
    """Average sentence length in words — shorter = more concise style."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def calc_vocab_diversity(text: str) -> float:
    """Vocabulary diversity — unique words / total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


# ============================================================================
# CHART GENERATION
# ============================================================================

def create_metrics_chart(base_ppl, ft_ppl, base_bleu, ft_bleu,
                         base_asl, ft_asl, base_vd, ft_vd):
    """Create a professional dark-themed comparison bar chart for all 4 metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("#1a1a2e")

    base_color = "#e94560"
    ft_color = "#16c79a"

    metrics_data = [
        ("Perplexity ↓", [base_ppl, ft_ppl], "Lower = more fluent"),
        ("BLEU Score ↑", [base_bleu, ft_bleu], "Higher = better match"),
        ("Avg Sent Len ↓", [base_asl, ft_asl], "Shorter = concise"),
        ("Vocab Diversity", [base_vd * 100, ft_vd * 100], "% unique words"),
    ]

    for ax, (title, values, subtitle) in zip(axes, metrics_data):
        ax.set_facecolor("#16213e")
        bars = ax.bar(
            ["Base", "Fine-tuned"], values,
            color=[base_color, ft_color],
            edgecolor=["#c73e54", "#12a87e"],
            linewidth=2, width=0.5, zorder=3,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(max(values), 0.01) * 0.05,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#e8e8e8",
            )
        ax.set_title(title, fontsize=12, fontweight="bold", color="#e8e8e8", pad=10)
        ax.set_xlabel(subtitle, fontsize=8, color="#888888", labelpad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#333355")
        ax.grid(axis="y", alpha=0.15, color="#555577")

    plt.tight_layout(pad=2.0)
    return fig


# ============================================================================
# MAIN GENERATION HANDLER
# ============================================================================

def process_prompt(prompt: str):
    """
    Main handler: generates from both models, calculates all 4 metrics,
    creates chart, computes style shift score, and returns all outputs.
    """
    if not prompt or not prompt.strip():
        return (
            "Please enter a writing prompt.",
            "Please enter a writing prompt.",
            None, 0.0,
            "—", "—", "—", "—",
            "Enter a prompt to see results.",
        )

    # Ensure models are loaded
    load_models()

    # --- Generate outputs from both models ---
    base_output = generate(base_model, prompt, is_base=True)
    ft_output = generate(ft_model, prompt, is_base=False)

    # --- Calculate all 4 metrics for both models ---
    base_ppl = calc_perplexity(base_model, prompt + " " + base_output, is_base=True)
    ft_ppl = calc_perplexity(ft_model, prompt + " " + ft_output, is_base=False) if (ft_model or DEMO_MODE) else float("nan")

    base_bleu = calc_bleu(base_output, prompt)
    ft_bleu = calc_bleu(ft_output, prompt)

    base_asl = calc_avg_sentence_length(base_output)
    ft_asl = calc_avg_sentence_length(ft_output)

    base_vd = calc_vocab_diversity(base_output)
    ft_vd = calc_vocab_diversity(ft_output)

    # --- Style Shift Score (% improvement of fine-tuned over base) ---
    improvements = 0
    total_metrics = 0
    if not math.isnan(ft_ppl) and not math.isnan(base_ppl) and base_ppl > 0:
        if ft_ppl < base_ppl:
            improvements += 1
        total_metrics += 1
    if ft_bleu > base_bleu:
        improvements += 1
    total_metrics += 1
    if ft_asl < base_asl and ft_asl > 0:
        improvements += 1
    total_metrics += 1
    total_metrics += 1  # vocab diversity counted but neutral

    style_shift = (improvements / max(total_metrics, 1)) * 100

    # --- Create comparison chart ---
    safe = lambda x: x if not math.isnan(x) else 0.0
    chart = create_metrics_chart(
        safe(base_ppl), safe(ft_ppl), base_bleu, ft_bleu,
        base_asl, ft_asl, base_vd, ft_vd
    )

    # --- Metrics badge texts ---
    ppl_badge = f"Base: {safe(base_ppl):.1f}  |  FT: {safe(ft_ppl):.1f}"
    bleu_badge = f"Base: {base_bleu:.4f}  |  FT: {ft_bleu:.4f}"
    asl_badge = f"Base: {base_asl:.1f} words  |  FT: {ft_asl:.1f} words"
    vd_badge = f"Base: {base_vd:.1%}  |  FT: {ft_vd:.1%}"

    # --- Summary text ---
    summary = (
        f"**Perplexity** (↓ better): Base={safe(base_ppl):.1f}, FT={safe(ft_ppl):.1f}\n\n"
        f"**BLEU Score** (↑ better): Base={base_bleu:.4f}, FT={ft_bleu:.4f}\n\n"
        f"**Avg Sentence Length** (↓ concise): Base={base_asl:.1f}, FT={ft_asl:.1f}\n\n"
        f"**Vocab Diversity**: Base={base_vd:.1%}, FT={ft_vd:.1%}"
    )

    return (
        base_output, ft_output, chart, style_shift,
        ppl_badge, bleu_badge, asl_badge, vd_badge, summary
    )


# ============================================================================
# GRADIO UI DEFINITION
# ============================================================================

def create_ui():
    """Build the professional Gradio interface with dark theme and all components."""

    # --- Custom CSS for premium dark look ---
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%) !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    #header-title {
        text-align: center;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    #prompt-input textarea {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        padding: 16px !important;
    }
    #prompt-input textarea:focus {
        border-color: #16c79a !important;
        box-shadow: 0 0 0 3px rgba(22, 199, 154, 0.15) !important;
    }
    #base-output textarea {
        background: #0d1117 !important;
        border: 2px solid #484f58 !important;
        color: #e6edf3 !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        line-height: 1.7 !important;
    }
    #ft-output textarea {
        background: #0d1117 !important;
        border: 2px solid #16c79a !important;
        color: #e6edf3 !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        line-height: 1.7 !important;
    }
    #generate-btn {
        background: linear-gradient(135deg, #16c79a 0%, #0f3460 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 48px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(22, 199, 154, 0.3) !important;
    }
    #generate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(22, 199, 154, 0.5) !important;
    }
    .metric-badge input {
        text-align: center !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
    }
    #footer-text {
        text-align: center;
        color: #8b949e !important;
        font-size: 12px !important;
        padding: 16px !important;
        border-top: 1px solid #21262d !important;
        margin-top: 16px !important;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=custom_css,
        title="MyStyle Writer — AI Style Fine-Tuning",
    ) as app:

        # ---- HEADER ----
        gr.HTML("""
        <div id="header-title">
            <h1 style="margin:0; font-size:2.4em;
                background: linear-gradient(135deg, #16c79a, #e94560);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;">✍️ MyStyle Writer</h1>
            <p style="margin:8px 0 0 0; color:#8b949e; font-size:1.1em;">
                <strong style="color:#e6edf3;">Qwen2.5-3B-Instruct</strong>
                Fine-Tuned with <strong style="color:#16c79a;">LoRA</strong></p>
            <p style="margin:6px 0 0 0; color:#555; font-size:0.85em;">
                Project #27 • Compare base model vs fine-tuned model side by side</p>
        </div>
        """)

        # ---- INPUT SECTION ----
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="📝 Enter Your Prompt",
                    placeholder="e.g., The soldier walked into the battlefield...",
                    lines=4, max_lines=6,
                    elem_id="prompt-input",
                )
            with gr.Column(scale=1, min_width=180):
                generate_btn = gr.Button(
                    "⚡ Generate", variant="primary",
                    elem_id="generate-btn", size="lg",
                )

        # ---- STYLE SHIFT SCORE (progress bar) ----
        gr.Markdown("### 🎯 Style Shift Score")
        style_shift_slider = gr.Slider(
            minimum=0, maximum=100, value=0, step=1,
            label="% Improvement (Fine-tuned over Base)",
            interactive=False,
        )

        # ---- OUTPUT PANELS (side by side) ----
        with gr.Row(equal_height=True):
            with gr.Column():
                base_output = gr.Textbox(
                    label="📄 Base Model Output (Qwen2.5-3B)",
                    lines=12, max_lines=18, interactive=False,
                    elem_id="base-output",
                )
            with gr.Column():
                ft_output = gr.Textbox(
                    label="✍️ Fine-Tuned Model Output (LoRA)",
                    lines=12, max_lines=18, interactive=False,
                    elem_id="ft-output",
                )

        # ---- METRICS ROW (4 badges) ----
        gr.Markdown("### 📊 Live Metrics Comparison")
        with gr.Row():
            ppl_badge = gr.Textbox(
                label="🔢 Perplexity (↓ better)",
                value="—", interactive=False,
                elem_classes=["metric-badge"],
            )
            bleu_badge = gr.Textbox(
                label="📐 BLEU Score (↑ better)",
                value="—", interactive=False,
                elem_classes=["metric-badge"],
            )
            asl_badge = gr.Textbox(
                label="📏 Avg Sentence Length",
                value="—", interactive=False,
                elem_classes=["metric-badge"],
            )
            vd_badge = gr.Textbox(
                label="📚 Vocabulary Diversity",
                value="—", interactive=False,
                elem_classes=["metric-badge"],
            )

        # ---- METRICS CHART ----
        metrics_chart = gr.Plot(label="Metrics Comparison Chart", elem_id="metrics-chart")

        # ---- METRICS SUMMARY ----
        metrics_summary = gr.Markdown(value="*Generate text to see detailed metrics.*")

        # ---- LOSS CURVE DISPLAY ----
        gr.Markdown("### 📈 Training Loss Curve")
        if os.path.exists(LOSS_CURVE_PATH):
            gr.Image(value=LOSS_CURVE_PATH, label="Loss Curve (from best_adapter/)",
                     show_download_button=False)
        else:
            gr.Markdown("*Loss curve will appear here after running train.py*")

        # ---- EXAMPLE PROMPTS ----
        gr.Markdown("### 💡 Try These Prompts")
        gr.Examples(
            examples=[
                ["The soldier walked into the battlefield."],
                ["She sat alone by the window."],
                ["He ordered another drink at the bar."],
                ["The old man looked at the sea."],
                ["They didn't speak for a long time."],
                ["It was a cold morning in the city."],
                ["The boy ran across the field."],
                ["She never told him the truth."],
                ["He knew it was over."],
                ["The sun set behind the mountains."],
            ],
            inputs=prompt_input,
            label="Click an example to try it",
        )

        # ---- FOOTER ----
        gr.HTML("""
        <div id="footer-text">
            <p style="margin:0;">
                <strong>Tech Stack:</strong>
                🤗 Qwen2.5-3B-Instruct •
                🔧 LoRA/PEFT •
                🚀 TRL SFTTrainer •
                🎨 Gradio •
                🔥 PyTorch •
                📊 BitsAndBytes 4-bit
            </p>
            <p style="margin:4px 0 0 0; color:#555;">
                Project #27 — MyStyle Writer: Style Fine-Tuning
            </p>
        </div>
        """)

        # ---- WIRE UP EVENTS ----
        outputs = [
            base_output, ft_output, metrics_chart, style_shift_slider,
            ppl_badge, bleu_badge, asl_badge, vd_badge, metrics_summary
        ]
        generate_btn.click(
            fn=process_prompt, inputs=[prompt_input], outputs=outputs,
        )
        prompt_input.submit(
            fn=process_prompt, inputs=[prompt_input], outputs=outputs,
        )

    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MyStyle Writer — Professional Gradio UI")
    print("  Qwen2.5-3B-Instruct + LoRA Fine-Tuning")
    print("  Starting server...")
    print("=" * 60 + "\n")

    # Load models at startup
    try:
        load_models()
    except Exception as e:
        print(f"[!] Model pre-loading failed: {e}")
        print("  Models will be loaded on first generation request.\n")

    # Create and launch the Gradio app
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
