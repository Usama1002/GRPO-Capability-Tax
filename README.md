# The GRPO Tax is Smaller Than You Think

Official code for the paper:

> **The GRPO Tax is Smaller Than You Think: A Longitudinal Study of Capability Preservation During Reasoning Training**

We study how GRPO training for mathematical reasoning affects non-target capabilities across five instruction-tuned models from four families. Using dense-checkpoint evaluation (every 50 training steps) on 13 benchmarks, we find that 85% of non-target capabilities remain within +/-2% of baseline after one epoch of LoRA-based GRPO training.

## Key Findings

- **85% preservation rate** across 60 model-benchmark evaluations (5 models x 12 non-target benchmarks)
- **Safety divergence**: Qwen and Phi safety remains stable; Gemma and Llama show directional degradation
- **GRPO vs SFT**: GRPO preserves 10/12 capabilities vs 8/12 for SFT with identical LoRA config
- **GRPO vs DPO**: DPO is more conservative (smaller capability shifts in both directions)
- **5,616 evaluation data points** across 432 checkpoint evaluations

## Released Artifacts

| Artifact | Link |
|----------|------|
| Qwen 1.5B GRPO adapter | [usama10/grpo-tax-qwen-1.5b](https://huggingface.co/usama10/grpo-tax-qwen-1.5b) |
| Qwen 3B GRPO adapter | [usama10/grpo-tax-qwen-3b](https://huggingface.co/usama10/grpo-tax-qwen-3b) |
| Phi 3.8B GRPO adapter | [usama10/grpo-tax-phi-3.8b](https://huggingface.co/usama10/grpo-tax-phi-3.8b) |
| Gemma 2B GRPO adapter | [usama10/grpo-tax-gemma-2b](https://huggingface.co/usama10/grpo-tax-gemma-2b) |
| Llama 3B GRPO adapter | [usama10/grpo-tax-llama-3b](https://huggingface.co/usama10/grpo-tax-llama-3b) |
| Qwen 1.5B DPO adapter | [usama10/grpo-tax-qwen-1.5b-dpo](https://huggingface.co/usama10/grpo-tax-qwen-1.5b-dpo) |
| Qwen 3B DPO adapter | [usama10/grpo-tax-qwen-3b-dpo](https://huggingface.co/usama10/grpo-tax-qwen-3b-dpo) |
| Full evaluation data | [usama10/grpo-tax-eval-data](https://huggingface.co/datasets/usama10/grpo-tax-eval-data) |

## Repository Structure

```
grpo-capability-tax/
  scripts/                       # Training and evaluation scripts
    train_grpo_checkpointed.py   # GRPO training with dense checkpoint saving
    train_dpo_checkpointed.py    # DPO training with dense checkpoints
    train_sft_checkpointed.py    # SFT baseline with dense checkpoints
    evaluate_checkpoint.py       # Evaluate one checkpoint on all 13 benchmarks
    run_experiment.py            # Orchestrator: train + evaluate all checkpoints
    run_dpo_experiment.py        # DPO experiment orchestrator
    reeval_new_benchmarks.py     # Add new benchmarks to existing eval data
    eval_flexible_gsm8k.py       # Strict vs flexible GSM8K answer extraction
  benchmarks/                    # 13 benchmark implementations
    base.py                      # Base class and model loading utilities
    math_bench.py                # GSM8K (target task)
    mmlu_bench.py                # MMLU
    commonsense_bench.py         # HellaSwag
    arc_bench.py                 # ARC-Challenge
    truthfulqa_bench.py          # TruthfulQA
    winogrande_bench.py          # Winogrande
    ifeval_bench.py              # Instruction following
    summarization_bench.py       # XSum (ROUGE-L)
    translation_bench.py         # WMT en->de (BLEU)
    coding_bench.py              # Python code generation
    safety_bench.py              # Safety refusal rate
    creative_bench.py            # Creative writing quality
    conversation_bench.py        # Conversational helpfulness
  analysis/
    generate_all_figures.py      # Generate all paper figures from eval data
  figures/                       # Generated figures
  results/                       # Evaluation results (download from HF)
```

## Setup

```bash
git clone https://github.com/usama10/grpo-capability-tax.git
cd grpo-capability-tax
pip install -r requirements.txt
```

## Reproducing Experiments

### 1. Run GRPO training with dense checkpoints

```bash
cd scripts
python train_grpo_checkpointed.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir ../results/qwen-1.5b/training \
    --checkpoint-interval 50
```

### 2. Run the full experiment pipeline (train + evaluate all checkpoints)

```bash
python run_experiment.py --models qwen-1.5b
```

This trains GRPO, saves checkpoints every 50 steps, then evaluates each checkpoint on all 13 benchmarks.

### 3. Run all five models

```bash
python run_experiment.py --models qwen-1.5b qwen-3b phi-3.8b gemma-2b llama-3b
```

### 4. Run the DPO comparison

```bash
python run_dpo_experiment.py --models qwen-1.5b qwen-3b
```

### 5. Run the SFT baseline

```bash
python train_sft_checkpointed.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir ../results/qwen-1.5b-sft/training \
    --checkpoint-interval 50
```

### 6. Generate figures

```bash
cd analysis
python generate_all_figures.py
```

## Evaluation Data

The full evaluation dataset (432 JSON files, 5,616 data points) is available on HuggingFace:

```bash
# Download using the HF CLI
hf download usama10/grpo-tax-eval-data --repo-type dataset --local-dir results/
```

Each JSON file contains scores on 13 benchmarks for a single model checkpoint:

```json
{
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "checkpoint": "base",
  "results": {
    "math_reasoning": {"score": 0.350, "num_examples": 200},
    "general_knowledge": {"score": 0.415, "num_examples": 200},
    ...
  }
}
```

## Models

| Model | Family | Params | Base |
|-------|--------|--------|------|
| Qwen2.5-1.5B-Instruct | Qwen | 1.5B | [HF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Qwen2.5-3B-Instruct | Qwen | 3.0B | [HF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| Phi-3.5-mini-instruct | Phi | 3.8B | [HF](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) |
| Gemma-2-2B-it | Gemma | 2.0B | [HF](https://huggingface.co/google/gemma-2-2b-it) |
| Llama-3.2-3B-Instruct | Llama | 3.2B | [HF](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |

All models trained with LoRA (r=16, alpha=32) on GSM8K for 1 epoch.

## Hardware

All experiments were run on a single NVIDIA RTX 5090 (32 GB VRAM). Total compute: approximately 100 GPU-hours.

## Citation

```bibtex
@article{usama2026grpotax,
  title={The GRPO Tax is Smaller Than You Think: A Longitudinal Study of Capability Preservation During Reasoning Training},
  author={Usama},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```

## License

Apache 2.0
