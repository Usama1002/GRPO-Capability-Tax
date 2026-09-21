#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
BASE="meta-llama/Llama-3.2-3B-Instruct"
OUT=results/llama-3b-5epoch
mkdir -p $OUT
echo "[$(date)] START Llama-3.2-3B 5-epoch GRPO on GSM8K"
$PY scripts/train_grpo_5epoch.py --model "$BASE" --output-dir $OUT/training --checkpoint-interval 50
echo "[$(date)] TRAIN DONE. Evaluating epoch boundaries."
$PY - <<'PY'
import json, glob, os
from benchmarks.base import load_model_from_checkpoint
from benchmarks.math_bench import MathBenchmark
from benchmarks.mmlu_bench import MMLUBenchmark
from benchmarks.ifeval_bench import IFEvalBenchmark
from benchmarks.creative_bench import CreativeWritingBenchmark
from benchmarks.summarization_bench import SummarizationBenchmark
from benchmarks.translation_bench import TranslationBenchmark
from benchmarks.coding_bench import CodingBenchmark
from benchmarks.safety_bench import SafetyBenchmark
from benchmarks.commonsense_bench import CommonsenseBenchmark
from benchmarks.conversation_bench import ConversationBenchmark
from benchmarks.arc_bench import ARCBenchmark
from benchmarks.truthfulqa_bench import TruthfulQABenchmark
from benchmarks.winogrande_bench import WinograndeBenchmark
BASE="meta-llama/Llama-3.2-3B-Instruct"; OUT="results/llama-3b-5epoch"
BENCHES=[MathBenchmark(num_examples=200),MMLUBenchmark(num_examples=200),IFEvalBenchmark(),
 CreativeWritingBenchmark(),SummarizationBenchmark(num_examples=100),TranslationBenchmark(num_examples=100),
 CodingBenchmark(),SafetyBenchmark(),CommonsenseBenchmark(num_examples=200),ConversationBenchmark(),
 ARCBenchmark(num_examples=200),TruthfulQABenchmark(num_examples=200),WinograndeBenchmark(num_examples=200)]
ckpts=sorted(glob.glob(OUT+"/training/checkpoint-*"), key=lambda p:int(p.split('-')[-1]))
steps=[int(p.split('-')[-1]) for p in ckpts]
boundaries={1:3736,2:7472,3:11208,4:14944,5:18680}
for ep,tgt in boundaries.items():
    near=min(ckpts, key=lambda p:abs(int(p.split('-')[-1])-tgt))
    print(f"[epoch {ep}] target {tgt} -> {near}", flush=True)
    model,tok=load_model_from_checkpoint(near, BASE)
    res={}
    for B in BENCHES:
        r=B.evaluate(model, tok); res[r['benchmark']]=r
        print(f"   {r['benchmark']}={r['score']:.3f}", flush=True)
    json.dump({"base_model":BASE,"checkpoint":near,"epoch":ep,"results":res}, open(f"{OUT}/eval_epoch_{ep}.json","w"), indent=2)
    del model
    import torch, gc; gc.collect(); torch.cuda.empty_cache()
print("ALL EPOCH EVALS DONE", flush=True)
PY
echo "[$(date)] ALL DONE"
