#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=results/qwen-1.5b-nonorm
mkdir -p $OUT
echo "[$(date)] START training (scale_rewards=False, G=4, 1 epoch)"
$PY scripts/train_grpo_nonorm.py --model Qwen/Qwen2.5-1.5B-Instruct --output-dir $OUT/training --checkpoint-interval 50
FINAL=$(ls -d $OUT/training/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "[$(date)] TRAIN DONE. final=$FINAL  Evaluating final on 10 benchmarks"
$PY scripts/evaluate_checkpoint.py --base-model Qwen/Qwen2.5-1.5B-Instruct --checkpoint "$FINAL" --output $OUT/eval_final.json
echo "[$(date)] Adding 3 new benchmarks (ARC/TruthfulQA/Winogrande)"
$PY scripts/reeval_new_benchmarks.py --models qwen-1.5b-nonorm || echo "reeval fallback may be needed"
echo "[$(date)] ALL DONE"
