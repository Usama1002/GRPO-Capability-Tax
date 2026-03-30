"""
Generate all publication-quality figures for the paper.
All figures follow strict readability guidelines:
  - Minimum 11pt axis labels, 10pt tick labels, 12pt subplot titles
  - White text on dark backgrounds, black on light
  - Single shared legends, no per-subplot repetition
  - No orphaned subplots, no fill on crowded charts
  - 300 DPI output
"""

import json
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Global font settings for publication readability
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.2,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
PAPER_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximally distinct colors (tested for color-blind friendliness)
MODEL_COLORS = {
    "qwen-1.5b": "#1f77b4",  # blue
    "qwen-3b": "#d62728",    # red
    "phi-3.8b": "#2ca02c",   # green
    "gemma-2b": "#ff7f0e",   # orange
    "llama-3b": "#9467bd",   # purple
}

MODEL_LABELS = {
    "qwen-1.5b": "Qwen 1.5B",
    "qwen-3b": "Qwen 3B",
    "phi-3.8b": "Phi 3.8B",
    "gemma-2b": "Gemma 2B",
    "llama-3b": "Llama 3B",
}

MODEL_MARKERS = {
    "qwen-1.5b": "o",
    "qwen-3b": "s",
    "phi-3.8b": "^",
    "gemma-2b": "D",
    "llama-3b": "v",
}

BENCHMARK_COLORS = {
    "math_reasoning": "#E53935", "general_knowledge": "#1E88E5",
    "instruction_following": "#43A047", "creative_writing": "#8E24AA",
    "summarization": "#F4511E", "translation": "#00ACC1",
    "coding": "#3949AB", "safety": "#C0CA33",
    "commonsense": "#6D4C41", "conversation": "#EC407A",
    "arc_challenge": "#FF6F00", "truthfulqa": "#00897B",
    "winogrande": "#5E35B1",
}

BENCHMARK_LABELS = {
    "math_reasoning": "Math (target)", "general_knowledge": "MMLU",
    "instruction_following": "IFEval", "creative_writing": "Creative Writing*",
    "summarization": "Summarization", "translation": "Translation",
    "coding": "Coding", "safety": "Safety",
    "commonsense": "HellaSwag", "conversation": "Conversation*",
    "arc_challenge": "ARC-Challenge", "truthfulqa": "TruthfulQA",
    "winogrande": "Winogrande",
}


def load_model_results(model_key):
    model_dir = os.path.join(RESULTS_DIR, model_key)
    evals = sorted(glob.glob(os.path.join(model_dir, "eval_*.json")))
    results = []
    for eval_path in evals:
        fname = os.path.basename(eval_path)
        step = 0 if fname == "eval_base.json" else int(fname.replace("eval_step_", "").replace(".json", ""))
        with open(eval_path) as f:
            data = json.load(f)
        entry = {"step": step}
        for bench_name, bench_result in data["results"].items():
            entry[bench_name] = bench_result["score"]
        results.append(entry)
    results.sort(key=lambda x: x["step"])
    return results


def moving_average(values, window=5):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    prefix = [np.mean(arr[:i+1]) for i in range(window - 1)]
    return np.concatenate([prefix, ma])


def save_fig(fig, name):
    """Save to both research/figures and paper/figures."""
    path1 = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    if os.path.exists(PAPER_FIG_DIR):
        path2 = os.path.join(PAPER_FIG_DIR, name)
        fig.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}")


# =========================================================================
# Figure 1: Individual model curves (3x5 grid, no orphan)
# =========================================================================
def plot_individual_curves():
    for model_key in ["qwen-1.5b", "qwen-3b", "phi-3.8b", "gemma-2b", "llama-3b"]:
        results = load_model_results(model_key)
        if len(results) < 3:
            continue

        steps = [r["step"] for r in results]
        benchmarks = [k for k in BENCHMARK_LABELS.keys() if k in results[0]]

        # Use 3x5 grid (15 slots for 13 benchmarks, hide 2)
        fig, axes = plt.subplots(3, 5, figsize=(24, 14))
        # No suptitle -- LaTeX caption handles the figure title

        for idx, bench in enumerate(benchmarks):
            row, col = idx // 5, idx % 5
            ax = axes[row, col]
            values = [r.get(bench, 0) for r in results]
            color = BENCHMARK_COLORS.get(bench, "#333")
            label = BENCHMARK_LABELS.get(bench, bench)

            # Only trend line (raw was too faint to be useful)
            trend = moving_average(values, window=5)
            ax.plot(steps, trend, color=color, linewidth=2.5)

            baseline = values[0]
            ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.4, linewidth=1)
            ax.set_title(label, fontsize=13, fontweight="bold")
            ax.set_xlabel("Step", fontsize=11)
            ax.set_ylabel("Score", fontsize=11)

            # Delta annotation -- large, readable, in consistent position
            delta = values[-1] - baseline
            sign = "+" if delta > 0 else ""
            c = "#2E7D32" if delta > 0.01 else "#C62828" if delta < -0.01 else "#555"
            ax.text(0.97, 0.08, f"{sign}{delta:.3f}", transform=ax.transAxes,
                    fontsize=12, ha="right", color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c, alpha=0.8))

        # Hide unused subplots
        for idx in range(len(benchmarks), 15):
            row, col = idx // 5, idx % 5
            axes[row, col].set_visible(False)

        plt.tight_layout()
        save_fig(fig, f"fig1_curves_{model_key}.png")


# =========================================================================
# Figure 2: Cross-model comparison (2x4, shared legend)
# =========================================================================
def plot_cross_model_comparison():
    models = {}
    for mk in ["qwen-1.5b", "qwen-3b", "phi-3.8b", "gemma-2b", "llama-3b"]:
        r = load_model_results(mk)
        if len(r) > 3:
            models[mk] = r

    key_benchmarks = ["math_reasoning", "safety", "arc_challenge", "truthfulqa",
                      "creative_writing", "commonsense", "coding", "translation"]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    # No suptitle -- LaTeX caption handles the figure title

    lines = []
    labels = []

    for idx, bench in enumerate(key_benchmarks):
        ax = axes[idx // 4, idx % 4]
        for mk, results in models.items():
            steps = [r["step"] for r in results]
            values = [r.get(bench, 0) for r in results]
            baseline = values[0]
            if baseline > 0:
                relative = [(v - baseline) / baseline * 100 for v in values]
            else:
                relative = [0] * len(values)
            trend = moving_average(relative, window=5)
            line, = ax.plot(steps, trend, color=MODEL_COLORS[mk], linewidth=2.5,
                           marker=MODEL_MARKERS[mk], markevery=15, markersize=5)
            if idx == 0:
                lines.append(line)
                labels.append(MODEL_LABELS[mk])

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(BENCHMARK_LABELS.get(bench, bench), fontsize=13, fontweight="bold")
        ax.set_xlabel("Step", fontsize=11)
        if idx % 4 == 0:
            ax.set_ylabel("Relative Change (%)", fontsize=11)

    # Single shared legend at the bottom, spanning full width
    leg = fig.legend(lines, labels, loc="lower center", ncol=5, fontsize=14,
               frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.01),
               handlelength=3, handleheight=1.5, handletextpad=1.0,
               columnspacing=3.0, borderpad=0.8, markerscale=1.5)
    leg.get_frame().set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    save_fig(fig, "fig2_cross_model_comparison.png")


# =========================================================================
# Figure 3: Onset heatmap (COMPLETELY REDONE)
# =========================================================================
def plot_onset_heatmap():
    models_order = ["qwen-1.5b", "qwen-3b", "phi-3.8b", "gemma-2b", "llama-3b"]
    models = {}
    for mk in models_order:
        r = load_model_results(mk)
        if len(r) > 3:
            models[mk] = r

    all_benchmarks = list(BENCHMARK_LABELS.keys())
    available = [b for b in all_benchmarks if all(b in models[mk][0] for mk in models)]

    onset_matrix = np.full((len(available), len(models_order)), np.nan)

    for j, mk in enumerate(models_order):
        results = models[mk]
        max_step = results[-1]["step"]
        for i, bench in enumerate(available):
            baseline = results[0].get(bench, 0)
            if baseline == 0:
                continue
            threshold = baseline * 0.98
            for r in results[1:]:
                idx = results.index(r)
                if idx >= 2:
                    avg = np.mean([results[k].get(bench, 0) for k in range(idx-2, idx+1)])
                else:
                    avg = r.get(bench, 0)
                if avg < threshold:
                    onset_matrix[i, j] = r["step"] / max_step * 100
                    break

    fig, ax = plt.subplots(figsize=(12, 11))

    # Use a yellow-orange-red palette for degradation, with light gray for stable
    display = np.where(np.isnan(onset_matrix), -1, onset_matrix)

    # Custom colormap: light yellow (early onset) -> dark red (late onset), gray for stable
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # Create a masked array approach instead
    import matplotlib.colors as mcolors

    # Plot with a warm sequential palette
    cmap = plt.cm.YlOrRd.copy()
    masked = np.ma.masked_where(np.isnan(onset_matrix), onset_matrix)

    # Fill background with light green for stable cells
    for i in range(len(available)):
        for j in range(len(models_order)):
            if np.isnan(onset_matrix[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor="#d4edda", edgecolor="white", linewidth=2))

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(models_order)))
    ax.set_xticklabels([MODEL_LABELS[mk] for mk in models_order], fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels([BENCHMARK_LABELS[b] for b in available], fontsize=12)

    # Add text with proper contrast
    for i in range(len(available)):
        for j in range(len(models_order)):
            val = onset_matrix[i, j]
            if np.isnan(val):
                text = "Stable"
                ax.text(j, i, text, ha="center", va="center", fontsize=11,
                        color="#155724", fontweight="bold")
            else:
                text = f"{val:.0f}%"
                # White text on dark cells, black on light
                text_color = "white" if val > 40 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=12,
                        color=text_color, fontweight="bold")

    # Add cell borders
    for i in range(len(available) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(len(models_order) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    # No title -- LaTeX caption handles this
    cbar = plt.colorbar(im, ax=ax, label="Training progress at onset (%)", shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_fig(fig, "fig3_onset_heatmap.png")


# =========================================================================
# Figure 4: GRPO vs DPO (reduced to 2x3, larger panels)
# =========================================================================
def plot_grpo_vs_dpo():
    pairs = [("qwen-1.5b", "qwen-1.5b-dpo"), ("qwen-3b", "qwen-3b-dpo")]

    key_benchmarks = ["math_reasoning", "safety", "creative_writing",
                      "commonsense", "coding", "summarization"]

    n_cols = len(key_benchmarks)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 10))
    # No suptitle -- LaTeX caption handles the figure title

    grpo_lines = []
    dpo_lines = []

    for row, (grpo_key, dpo_key) in enumerate(pairs):
        grpo_results = load_model_results(grpo_key)
        dpo_results = load_model_results(dpo_key)
        model_label = MODEL_LABELS[grpo_key]

        for col, bench in enumerate(key_benchmarks):
            ax = axes[row, col]

            # GRPO curve
            grpo_steps = [r["step"] for r in grpo_results]
            grpo_vals = [r.get(bench, 0) for r in grpo_results]
            grpo_baseline = grpo_vals[0]
            if grpo_baseline > 0:
                grpo_rel = [(v - grpo_baseline) / grpo_baseline * 100 for v in grpo_vals]
            else:
                grpo_rel = [0] * len(grpo_vals)
            grpo_trend = moving_average(grpo_rel, window=5)
            l1, = ax.plot(grpo_steps, grpo_trend, color="#d62728", linewidth=2.5, label="GRPO")

            # DPO curve
            dpo_steps = [r["step"] for r in dpo_results]
            dpo_vals = [r.get(bench, 0) for r in dpo_results]
            dpo_baseline = dpo_vals[0]
            if dpo_baseline > 0:
                dpo_rel = [(v - dpo_baseline) / dpo_baseline * 100 for v in dpo_vals]
            else:
                dpo_rel = [0] * len(dpo_vals)
            dpo_trend = moving_average(dpo_rel, window=5)
            l2, = ax.plot(dpo_steps, dpo_trend, color="#1f77b4", linewidth=2.5,
                         linestyle="--", label="DPO")

            if row == 0 and col == 0:
                grpo_lines = [l1]
                dpo_lines = [l2]

            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
            title = BENCHMARK_LABELS.get(bench, bench)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Step", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{model_label}\nRelative Change (%)", fontsize=11)

    # Shared legend, large and spanning width
    leg = fig.legend(grpo_lines + dpo_lines, ["GRPO (RL on GSM8K)", "DPO (preference on UltraFeedback)"],
               loc="lower center", ncol=2, fontsize=14, frameon=True,
               bbox_to_anchor=(0.5, -0.01),
               handlelength=4, handleheight=1.5, handletextpad=1.0,
               columnspacing=4.0, borderpad=0.8)
    leg.get_frame().set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    save_fig(fig, "fig4_grpo_vs_dpo.png")


# =========================================================================
# Figure 5: Radar chart (cleaner, no fill, larger labels)
# =========================================================================
def plot_radar_chart():
    models_order = ["qwen-1.5b", "qwen-3b", "phi-3.8b", "gemma-2b", "llama-3b"]
    benchmarks = ["arc_challenge", "truthfulqa", "winogrande", "commonsense",
                  "general_knowledge", "instruction_following", "safety",
                  "coding", "summarization", "translation"]

    retention = {}
    for mk in models_order:
        results = load_model_results(mk)
        if len(results) < 2:
            continue
        base = results[0]
        final = results[-1]
        ratios = []
        for bench in benchmarks:
            b = base.get(bench, 0)
            f = final.get(bench, 0)
            ratio = f / b if b > 0 else 1.0
            ratios.append(min(ratio, 1.2))
        retention[mk] = ratios

    N = len(benchmarks)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    # No title -- LaTeX caption handles this

    for mk in models_order:
        if mk not in retention:
            continue
        values = retention[mk] + retention[mk][:1]
        ax.plot(angles, values, color=MODEL_COLORS[mk], linewidth=2.5,
                label=MODEL_LABELS[mk], marker=MODEL_MARKERS[mk], markersize=6)
        # No fill -- too cluttered with 5 models

    # Reference circle at 1.0
    ref = [1.0] * (N + 1)
    ax.plot(angles, ref, color="gray", linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([BENCHMARK_LABELS.get(b, b) for b in benchmarks], fontsize=11)
    ax.set_ylim(0.85, 1.15)
    ax.set_yticks([0.9, 0.95, 1.0, 1.05, 1.1])
    ax.set_yticklabels(["0.90", "0.95", "1.00", "1.05", "1.10"], fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11, frameon=True)
    plt.tight_layout()
    save_fig(fig, "fig5_radar_retention.png")


# =========================================================================
# Figure 6: Delta heatmap (minor fixes)
# =========================================================================
def plot_delta_heatmap():
    models_order = ["qwen-1.5b", "qwen-3b", "phi-3.8b", "gemma-2b", "llama-3b"]
    benchmarks = list(BENCHMARK_LABELS.keys())

    delta_matrix = np.zeros((len(benchmarks), len(models_order)))
    for j, mk in enumerate(models_order):
        results = load_model_results(mk)
        if len(results) < 2:
            continue
        base = results[0]
        final = results[-1]
        for i, bench in enumerate(benchmarks):
            b = base.get(bench, 0)
            f = final.get(bench, 0)
            delta_matrix[i, j] = f - b

    fig, ax = plt.subplots(figsize=(11, 11))
    im = ax.imshow(delta_matrix, cmap="RdBu", aspect="auto", vmin=-0.08, vmax=0.08)

    ax.set_xticks(range(len(models_order)))
    ax.set_xticklabels([MODEL_LABELS[mk] for mk in models_order], fontsize=13, fontweight="bold")
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels([BENCHMARK_LABELS[b] for b in benchmarks], fontsize=12)

    for i in range(len(benchmarks)):
        for j in range(len(models_order)):
            val = delta_matrix[i, j]
            # Fix: use "0.000" instead of "+0.000" or "-0.000"
            if abs(val) < 0.0005:
                text = "0.000"
            else:
                sign = "+" if val > 0 else ""
                text = f"{sign}{val:.3f}"
            c = "white" if abs(val) > 0.04 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color=c, fontweight="bold")

    # Cell borders
    for i in range(len(benchmarks) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(len(models_order) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    # No title -- LaTeX caption handles this
    cbar = plt.colorbar(im, ax=ax, label="Score change (final - baseline)", shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    save_fig(fig, "fig6_delta_heatmap.png")


# =========================================================================
# Main
# =========================================================================
def main():
    print("Generating publication-quality figures...\n")
    plot_individual_curves()
    plot_cross_model_comparison()
    plot_onset_heatmap()
    plot_grpo_vs_dpo()
    plot_radar_chart()
    plot_delta_heatmap()
    print(f"\nAll figures saved to {OUTPUT_DIR} and {PAPER_FIG_DIR}")


if __name__ == "__main__":
    main()
