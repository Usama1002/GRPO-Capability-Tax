# The GRPO Tax is Smaller Than You Think

Code and data for reproducing all experiments in:

> **The GRPO Tax is Smaller Than You Think: A Longitudinal Study of Capability Preservation During Reasoning Training**
> Muhammad Usama. *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=e0UVcimXdK)

We study how GRPO training for mathematical reasoning affects non-target capabilities across five instruction-tuned models from four families. Using dense-checkpoint evaluation (every 50 training steps) on 13 benchmarks, we find that 85% of non-target capabilities remain within +/-2% of baseline after one epoch of LoRA-based GRPO training, 8% improve, and 7% degrade.

## Released Artifacts

All adapters and evaluation data are on the Hugging Face Hub.

| Artifact | Type | Base model | Link |
|---|---|---|---|
| Qwen 1.5B GRPO | LoRA adapter | Qwen2.5-1.5B-Instruct | [usama10/grpo-tax-qwen-1.5b](https://huggingface.co/usama10/grpo-tax-qwen-1.5b) |
| Qwen 3B GRPO | LoRA adapter | Qwen2.5-3B-Instruct | [usama10/grpo-tax-qwen-3b](https://huggingface.co/usama10/grpo-tax-qwen-3b) |
| Phi 3.8B GRPO | LoRA adapter | Phi-3.5-mini-instruct | [usama10/grpo-tax-phi-3.8b](https://huggingface.co/usama10/grpo-tax-phi-3.8b) |
| Gemma 2B GRPO | LoRA adapter | Gemma-2-2B-it | [usama10/grpo-tax-gemma-2b](https://huggingface.co/usama10/grpo-tax-gemma-2b) |
| Llama 3B GRPO | LoRA adapter | Llama-3.2-3B-Instruct | [usama10/grpo-tax-llama-3b](https://huggingface.co/usama10/grpo-tax-llama-3b) |
| Qwen 1.5B DPO | LoRA adapter | Qwen2.5-1.5B-Instruct | [usama10/grpo-tax-qwen-1.5b-dpo](https://huggingface.co/usama10/grpo-tax-qwen-1.5b-dpo) |
| Qwen 3B DPO | LoRA adapter | Qwen2.5-3B-Instruct | [usama10/grpo-tax-qwen-3b-dpo](https://huggingface.co/usama10/grpo-tax-qwen-3b-dpo) |
| Evaluation data | 432 JSON files, 5,616 scores | - | [usama10/grpo-tax-eval-data](https://huggingface.co/usama10/grpo-tax-eval-data) |

## Setup

```bash
git clone https://github.com/Usama1002/GRPO-Capability-Tax.git
cd GRPO-Capability-Tax
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, PyTorch 2.7+, CUDA 12.x, single GPU with 32 GB VRAM (tested on RTX 5090). Gated models (Gemma, Llama) require Hugging Face access approval before training.

All commands below are run from the repository root.

## Reproducing the Main Results

The pipeline for each model is three phases: GRPO training with dense checkpointing, evaluation of every checkpoint on 13 benchmarks, and figure generation. The orchestrator handles the first two.

```bash
# Verify the setup (2 training steps, ~5 minutes)
python scripts/run_experiment.py --models qwen-1.5b --dry-run

# One model end-to-end (~10-14 hours for the small models)
python scripts/run_experiment.py --models qwen-1.5b

# All five GRPO models
python scripts/run_experiment.py --models qwen-1.5b qwen-3b phi-3.8b gemma-2b llama-3b

# DPO comparison (Section 5.7)
python scripts/run_dpo_experiment.py --models qwen-1.5b qwen-3b

# Strict vs flexible GSM8K extraction (Section 6)
python scripts/eval_flexible_gsm8k.py --models qwen-1.5b
```

SFT baseline (Section 5.8):

```bash
python scripts/train_sft_checkpointed.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir results/qwen-1.5b-sft/training \
    --checkpoint-interval 50

python scripts/evaluate_checkpoint.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --checkpoint results/qwen-1.5b-sft/training/checkpoint-935 \
    --output results/qwen-1.5b-sft/eval_final.json
```

## Reproducing the Supplementary Experiments

Group-normalization ablation (Section 5.10). Disables the group-relative standard-deviation scaling, so advantages are mean-centred only:

```bash
bash scripts/run_nonorm_ablation.sh
```

Five-epoch continuation (Section 5.13). The runner trains Llama-3.2-3B for five epochs and evaluates each epoch boundary:

```bash
bash scripts/run_llama_5epoch.sh

# Or train any model for five epochs directly
python scripts/train_grpo_5epoch.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir results/qwen-1.5b-5epoch/training \
    --checkpoint-interval 50
```

Per-benchmark KL drift, used to validate the capability-drift bound (Sections 5.14 and 5.17):

```bash
python scripts/measure_kl_drift.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --checkpoint results/qwen-1.5b/training/checkpoint-3736 \
    --benchmarks math_reasoning general_knowledge commonsense safety \
    --num_prompts 100 --num_samples 4 \
    --output results/qwen-1.5b/kl_drift.json
```

Both shell runners honour a `PYTHON` environment variable if you are not using the default interpreter:

```bash
PYTHON=/path/to/env/bin/python bash scripts/run_nonorm_ablation.sh
```

## Figures

To regenerate figures without rerunning training, download the released evaluation data:

```bash
pip install huggingface_hub
hf download usama10/grpo-tax-eval-data --local-dir results/
python analysis/generate_all_figures.py
```

Each JSON file holds scores on 13 benchmarks for a single checkpoint:

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
GRPO-Capability-Tax/
  scripts/
    run_experiment.py             # Orchestrator: GRPO train + evaluate all checkpoints
    run_dpo_experiment.py         # DPO experiment orchestrator
    train_grpo_checkpointed.py    # GRPO training with dense checkpoint saving
    train_dpo_checkpointed.py     # DPO training with dense checkpoints
    train_sft_checkpointed.py     # SFT baseline with dense checkpoints
    train_grpo_nonorm.py          # GRPO with group-std normalization disabled (Section 5.10)
    train_grpo_5epoch.py          # GRPO for five epochs (Section 5.13)
    evaluate_checkpoint.py        # Evaluate one checkpoint on all 13 benchmarks
    reeval_new_benchmarks.py      # Add new benchmarks to existing eval data
    eval_flexible_gsm8k.py        # Strict vs flexible GSM8K extraction comparison
    measure_kl_drift.py           # Per-benchmark sequence-level KL from the base policy
    run_nonorm_ablation.sh        # Runner: group-normalization ablation
    run_llama_5epoch.sh           # Runner: Llama-3.2-3B five-epoch continuation
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

All models use identical hyperparameters.

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

## Citation

```bibtex
@article{usama2026grpotax,
  title   = {The {GRPO} Tax is Smaller Than You Think: A Longitudinal Study of Capability Preservation During Reasoning Training},
  author  = {Muhammad Usama},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=e0UVcimXdK}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
