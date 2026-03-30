# The GRPO Tax is Smaller Than You Think

Code and data for reproducing all experiments in the paper:

> **The GRPO Tax is Smaller Than You Think: A Longitudinal Study of Capability Preservation During Reasoning Training**

We study how GRPO training for mathematical reasoning affects non-target capabilities across five instruction-tuned models from four families. Using dense-checkpoint evaluation (every 50 training steps) on 13 benchmarks, we find that 85% of non-target capabilities remain within +/-2% of baseline after one epoch of LoRA-based GRPO training.

## Setup

```bash
git clone <this-repo-url>
cd grpo-capability-tax
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, PyTorch 2.7+, CUDA 12.x, single GPU with 32 GB VRAM (tested on RTX 5090). Gated models (Gemma, Llama) require HuggingFace access approval before training.

## Reproducing All Results

The full pipeline for each model consists of three phases: (1) GRPO training with dense checkpointing, (2) evaluation of every checkpoint on 13 benchmarks, and (3) figure generation. The orchestrator script handles phases 1 and 2 automatically.

### Quick start: reproduce one model end-to-end

```bash
cd scripts

# Train GRPO + evaluate all checkpoints (takes ~10-14 hours for small models)
python run_experiment.py --models qwen-1.5b

# Dry-run to verify setup (2 training steps only, ~5 minutes)
python run_experiment.py --models qwen-1.5b --dry-run
```

### Reproduce all five GRPO models

```bash
# Sequential execution (each model: ~10-14 hours training + evaluation)
python run_experiment.py --models qwen-1.5b qwen-3b phi-3.8b gemma-2b llama-3b
```

### Reproduce the DPO comparison (Qwen 1.5B and 3B)

```bash
python run_dpo_experiment.py --models qwen-1.5b qwen-3b
```

### Reproduce the SFT baseline (Qwen 1.5B)

```bash
python train_sft_checkpointed.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir ../results/qwen-1.5b-sft/training \
    --checkpoint-interval 50

# Then evaluate the final checkpoint
python evaluate_checkpoint.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --checkpoint ../results/qwen-1.5b-sft/training/checkpoint-935 \
    --output ../results/qwen-1.5b-sft/eval_final.json
```

### Reproduce the flexible vs strict GSM8K evaluation

```bash
python eval_flexible_gsm8k.py --models qwen-1.5b
```

### Generate all figures from evaluation data

```bash
cd analysis
python generate_all_figures.py
# Output saved to ../figures/
```

## Evaluation Data

All 432 evaluation files (5,616 data points) are available for download. To reproduce figures and analysis without rerunning training:

```bash
# Download pre-computed evaluation results
pip install huggingface_hub
huggingface-cli download usama10/grpo-tax-eval-data --repo-type dataset --local-dir results/

# Generate figures from downloaded data
cd analysis
python generate_all_figures.py
```

Each JSON file contains scores on 13 benchmarks for a single model checkpoint:

```json
{
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "checkpoint": "results/qwen-1.5b/training/checkpoint-3736",
  "results": {
    "math_reasoning": {"benchmark": "math_reasoning", "metric": "accuracy", "score": 0.355, "num_examples": 200},
    "general_knowledge": {"benchmark": "general_knowledge", "metric": "accuracy", "score": 0.425, "num_examples": 200},
    "commonsense": {"benchmark": "commonsense", "metric": "accuracy", "score": 0.550, "num_examples": 200},
    "arc_challenge": {"benchmark": "arc_challenge", "metric": "accuracy", "score": 0.715, "num_examples": 200},
    "truthfulqa": {"benchmark": "truthfulqa", "metric": "accuracy", "score": 0.360, "num_examples": 200},
    "winogrande": {"benchmark": "winogrande", "metric": "accuracy", "score": 0.545, "num_examples": 200},
    "instruction_following": {"benchmark": "instruction_following", "metric": "constraint_satisfaction", "score": 0.844, "num_examples": 30},
    "summarization": {"benchmark": "summarization", "metric": "rouge_l", "score": 0.171, "num_examples": 66},
    "translation": {"benchmark": "translation", "metric": "bleu", "score": 0.058, "num_examples": 97},
    "coding": {"benchmark": "coding", "metric": "code_quality", "score": 0.984, "num_examples": 25},
    "safety": {"benchmark": "safety", "metric": "refusal_rate", "score": 0.967, "num_examples": 30},
    "creative_writing": {"benchmark": "creative_writing", "metric": "quality_score", "score": 0.693, "num_examples": 25},
    "conversation": {"benchmark": "conversation", "metric": "helpfulness", "score": 0.682, "num_examples": 25}
  }
}
```

## Repository Structure

```
grpo-capability-tax/
  scripts/
    train_grpo_checkpointed.py    # GRPO training with dense checkpoint saving
    train_dpo_checkpointed.py     # DPO training with dense checkpoints
    train_sft_checkpointed.py     # SFT baseline with dense checkpoints
    evaluate_checkpoint.py        # Evaluate one checkpoint on all 13 benchmarks
    run_experiment.py             # Orchestrator: train + evaluate all checkpoints
    run_dpo_experiment.py         # DPO experiment orchestrator
    reeval_new_benchmarks.py      # Add new benchmarks to existing eval data
    eval_flexible_gsm8k.py        # Strict vs flexible GSM8K extraction comparison
  benchmarks/
    base.py                       # Base benchmark class and model loading
    math_bench.py                 # GSM8K accuracy (target task, N=200)
    mmlu_bench.py                 # MMLU accuracy (N=200)
    commonsense_bench.py          # HellaSwag accuracy (N=200)
    arc_bench.py                  # ARC-Challenge accuracy (N=200)
    truthfulqa_bench.py           # TruthfulQA accuracy (N=200)
    winogrande_bench.py           # Winogrande accuracy (N=200)
    ifeval_bench.py               # Instruction following, constraint satisfaction (N=30)
    summarization_bench.py        # XSum ROUGE-L (N=66)
    translation_bench.py          # WMT en->de BLEU (N=97)
    coding_bench.py               # Python syntax + structure scoring (N=25)
    safety_bench.py               # Safety refusal rate via keyword matching (N=30)
    creative_bench.py             # Vocabulary diversity + richness scoring (N=25)
    conversation_bench.py         # Helpfulness heuristic scoring (N=25)
  analysis/
    generate_all_figures.py       # Generate all figures from eval data
  figures/                        # Pre-generated analysis figures
  results/                        # Evaluation results (download from HF or generate)
```

## Training Configuration

All models use identical hyperparameters:

| Parameter | GRPO | DPO | SFT |
|-----------|------|-----|-----|
| Dataset | openai/gsm8k (7,473) | UltraFeedback (10K) | openai/gsm8k (7,473) |
| Epochs | 1 | 1 | 1 |
| Learning rate | 5e-6 | 5e-7 | 5e-6 |
| LR scheduler | Cosine | Cosine | Cosine |
| Warmup steps | 20 | 20 | 20 |
| Batch size (effective) | 8 | 8 | 8 |
| LoRA rank | 16 | 16 | 16 |
| LoRA alpha | 32 | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 | 0.05 |
| LoRA targets | q,k,v,o,gate,up,down | q,k,v,o,gate,up,down | q,k,v,o,gate,up,down |
| Precision | bf16 | bf16 | bf16 |
| Checkpoint interval | 50 steps | 50 steps | 50 steps |
| GRPO group size | 4 | - | - |
| DPO beta | - | 0.1 | - |
| Max completion length | 512 | 768 | 1024 |
| Random seed | 42 | 42 | 42 |

## Models

| Model | Family | Params |
|-------|--------|--------|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Qwen | 1.5B |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Qwen | 3.0B |
| [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | Phi | 3.8B |
| [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it) | Gemma | 2.0B |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Llama | 3.2B |

## Expected Runtime

On a single NVIDIA RTX 5090 (32 GB VRAM):

| Experiment | Training | Evaluation | Total |
|-----------|----------|------------|-------|
| One GRPO model | 2-8 hours | 5-8 hours | 7-16 hours |
| All 5 GRPO models | 15-30 hours | 25-40 hours | 40-70 hours |
| DPO (2 models) | 4-6 hours | 4-6 hours | 8-12 hours |
| SFT baseline (1 model) | 1-2 hours | 10 min | ~2 hours |
| Flexible GSM8K eval | - | 30 min | 30 min |
| Figure generation | - | 2 min | 2 min |
| **Full reproduction** | - | - | **~50-85 hours** |

## License

Apache 2.0
