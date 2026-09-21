"""
Measure per-benchmark policy KL drift for the capability-drift bound (Eq. kl_bound).

For a trained policy pi_theta (LoRA checkpoint merged into base) and the reference
pi_ref (base instruct model), this estimates, per benchmark k:

    Dbar_KL,k = E_{x~D_k} D_KL( pi_theta(.|x) || pi_ref(.|x) )

using the single-sample Monte Carlo (k1) estimator with y sampled FROM pi_theta
at temperature 1.0 (full distribution, no top-k/top-p truncation), so the estimate
is unbiased for the FORWARD KL that appears in Pinsker's inequality with P=pi_theta:

    per-sample contribution = sum_t [ log pi_theta(y_t | x, y_<t) - log pi_ref(y_t | x, y_<t) ]

We report the mean sequence-level KL, its standard error, the mean completion
length, and the Pinsker bound sqrt(Dbar_KL,k / 2), which upper-bounds |Delta_k|
for any benchmark metric in [0, 1].

IMPORTANT CAVEATS (read before using numbers in the paper):
  1. The paper's reported Delta_k use GREEDY decoding (temperature 0). The KL here
     is over the temperature-1 sampling distribution. The bound |Delta_k| <= sqrt(KL/2)
     is stated for the sampled-policy scores; comparing to greedy Delta_k is therefore
     a heuristic check, not a strict inequality test. To test the strict bound, also
     measure Delta_k under temperature-1 sampling (not done here).
  2. The k1 estimator is unbiased but high variance per sample; we average over
     (prompts x samples). Increase --num_samples / --num_prompts to tighten the SE.
  3. Sequence-level KL grows with completion length; we also report per-token KL.

Usage (run inside the hf-training conda env):
  python measure_kl_drift.py \
      --base_model Qwen/Qwen2.5-1.5B-Instruct \
      --checkpoint results/qwen-1.5b/training/checkpoint-3736 \
      --benchmarks creative_writing instruction_following safety coding conversation \
      --num_prompts 40 --num_samples 4 --max_new_tokens 200 \
      --output results/qwen-1.5b/kl_drift_step3736.json
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Make the repository root importable so 'benchmarks' resolves from any directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.math_bench import MathBenchmark
from benchmarks.mmlu_bench import MMLUBenchmark
from benchmarks.arc_bench import ARCBenchmark
from benchmarks.winogrande_bench import WinograndeBenchmark
from benchmarks.truthfulqa_bench import TruthfulQABenchmark
from benchmarks.commonsense_bench import CommonsenseBenchmark
from benchmarks.ifeval_bench import IFEvalBenchmark
from benchmarks.creative_bench import CreativeWritingBenchmark
from benchmarks.coding_bench import CodingBenchmark
from benchmarks.safety_bench import SafetyBenchmark
from benchmarks.conversation_bench import ConversationBenchmark
from benchmarks.summarization_bench import SummarizationBenchmark
from benchmarks.translation_bench import TranslationBenchmark

BENCH_REGISTRY = {
    "math_reasoning": lambda: MathBenchmark(num_examples=200),
    "general_knowledge": lambda: MMLUBenchmark(num_examples=200),
    "arc_challenge": lambda: ARCBenchmark(num_examples=200),
    "winogrande": lambda: WinograndeBenchmark(num_examples=200),
    "truthfulqa": lambda: TruthfulQABenchmark(num_examples=200),
    "commonsense": lambda: CommonsenseBenchmark(num_examples=200),
    "instruction_following": lambda: IFEvalBenchmark(),
    "creative_writing": lambda: CreativeWritingBenchmark(),
    "coding": lambda: CodingBenchmark(),
    "safety": lambda: SafetyBenchmark(),
    "conversation": lambda: ConversationBenchmark(),
    "summarization": lambda: SummarizationBenchmark(num_examples=66),
    "translation": lambda: TranslationBenchmark(num_examples=97),
}


def load_model(base_model_name, checkpoint_path):
    if checkpoint_path == "base":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def build_prompt_text(benchmark, example, tokenizer, supports_system):
    messages = benchmark.format_prompt(example)
    if not supports_system and messages and messages[0]["role"] == "system":
        sys_content = messages[0]["content"]
        messages = messages[1:]
        if messages:
            messages[0]["content"] = sys_content + "\n\n" + messages[0]["content"]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def seq_logprob(model, full_ids, prompt_len):
    """Sum log p(y_t | x, y_<t) over the completion tokens of a single sequence."""
    out = model(full_ids)
    logits = out.logits[0, :-1, :]            # predict token t+1 from position t
    targets = full_ids[0, 1:]                 # next-token targets
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logprobs[torch.arange(targets.shape[0]), targets]
    comp_lp = tok_lp[prompt_len - 1:]         # completion tokens only
    return comp_lp.sum().item(), comp_lp.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--num_prompts", type=int, default=40)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    supports_system = True
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "t"}, {"role": "user", "content": "t"}],
            tokenize=False,
        )
    except Exception:
        supports_system = False

    print("loading policy (trained) ...", flush=True)
    policy = load_model(args.base_model, args.checkpoint)
    print("loading reference (base) ...", flush=True)
    ref = load_model(args.base_model, "base")

    results = {}
    for bname in args.benchmarks:
        if bname not in BENCH_REGISTRY:
            print(f"  [skip] unknown benchmark {bname}")
            continue
        bench = BENCH_REGISTRY[bname]()
        examples = bench.get_examples()[: args.num_prompts]
        per_seq_kl = []
        per_tok_kl = []
        t0 = time.time()
        for ex in examples:
            text = build_prompt_text(bench, ex, tokenizer, supports_system)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(policy.device)
            prompt_len = enc.input_ids.shape[1]
            gen = policy.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,                       # no truncation: sample from full distribution
                repetition_penalty=1.0,        # CRITICAL: must equal scoring distribution
                no_repeat_ngram_size=0,        # no n-gram blocking
                renormalize_logits=False,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
            for s in range(gen.shape[0]):
                comp = gen[s][prompt_len:]
                # Trim trailing padding: keep up to and including the first EOS,
                # drop the batch padding that generate() appends to shorter samples.
                eos_pos = (comp == tokenizer.eos_token_id).nonzero()
                end = (eos_pos[0].item() + 1) if eos_pos.numel() > 0 else comp.shape[0]
                if end == 0:
                    continue
                full = torch.cat([gen[s][:prompt_len], comp[:end]]).unsqueeze(0)
                if full.shape[1] <= prompt_len:
                    continue
                lp_pi, n_tok = seq_logprob(policy, full, prompt_len)
                lp_ref, _ = seq_logprob(ref, full, prompt_len)
                kl = lp_pi - lp_ref            # forward-KL k1 estimator, y ~ pi_theta
                per_seq_kl.append(kl)
                if n_tok > 0:
                    per_tok_kl.append(kl / n_tok)
        n = len(per_seq_kl)
        mean_kl = sum(per_seq_kl) / n if n else float("nan")
        var = sum((x - mean_kl) ** 2 for x in per_seq_kl) / (n - 1) if n > 1 else float("nan")
        se = (var / n) ** 0.5 if n > 1 else float("nan")
        mean_tok = sum(per_tok_kl) / len(per_tok_kl) if per_tok_kl else float("nan")
        # Pinsker uses non-negative KL; clamp the noisy MC mean at 0 for the bound.
        pinsker = (max(mean_kl, 0.0) / 2.0) ** 0.5
        results[bname] = {
            "mean_seq_kl": mean_kl,
            "se_seq_kl": se,
            "mean_token_kl": mean_tok,
            "pinsker_bound_on_abs_delta": pinsker,
            "n_samples": n,
            "num_prompts": len(examples),
            "samples_per_prompt": args.num_samples,
            "seconds": time.time() - t0,
        }
        print(
            f"  {bname:24} seqKL={mean_kl:8.3f} (+/-{se:5.3f})  tokKL={mean_tok:6.4f}  "
            f"Pinsker|Δ|<= {pinsker:6.3f}  n={n}  {results[bname]['seconds']:.0f}s",
            flush=True,
        )

    payload = {
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "estimator": "forward KL D_KL(pi_theta||pi_ref), k1, y~pi_theta @ temp=1.0, top_k=0",
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
