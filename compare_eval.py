import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_overall_metrics(sum1, sum2, name1, name2, out_dir: Path):
    metrics = [
        "overall_accuracy",
        "difficulty_weighted_accuracy",
        "dimension_weighted_accuracy",
        "combined_weighted_accuracy",
    ]
    labels = [
        "overall",
        "difficulty_w",
        "dimension_w",
        "combined_w",
    ]

    vals1 = [sum1[m] for m in metrics]
    vals2 = [sum2[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 4))
    bars1 = plt.bar(x - width / 2, vals1, width, label=name1)
    bars2 = plt.bar(x + width / 2, vals2, width, label=name2)

    # 在柱子上标数字（保留两位小数）
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    plt.xticks(x, labels, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Overall metrics comparison")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "overall_metrics.png"
    plt.savefig(out_path, dpi=200)
    print(f"saved {out_path}")
    plt.close()


def plot_per_dimension(sum1, sum2, name1, name2, out_dir: Path):
    dims1 = sum1["per_dimension"]
    dims2 = sum2["per_dimension"]

    # 取两个模型公共的 dimension，按字典序排一下，方便对齐
    common_dims = sorted(set(dims1.keys()) & set(dims2.keys()))

    x = np.arange(len(common_dims))
    width = 0.35

    vals1 = [dims1[d]["accuracy"] for d in common_dims]
    vals2 = [dims2[d]["accuracy"] for d in common_dims]

    plt.figure(figsize=(max(10, len(common_dims) * 0.5), 5))
    bars1 = plt.bar(x - width / 2, vals1, width, label=name1)
    bars2 = plt.bar(x + width / 2, vals2, width, label=name2)

    # 在柱子上标数字（保留两位小数）
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90 if len(common_dims) > 15 else 0,
        )

    plt.xticks(x, common_dims, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Per-dimension accuracy comparison")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "per_dimension_accuracy.png"
    plt.savefig(out_path, dpi=200)
    print(f"saved {out_path}")
    plt.close()


def main():
    path1 = Path("/home/siweim/code/PhyMetric/output100/Wan2.1-T2V-14B/baseline/eval_summary.json")
    path2 = Path("/home/siweim/code/PhyMetric/output100/Wan2.1-T2V-CoT-QwenVL/eval_summary.json")

    sum1 = load_summary(path1)
    sum2 = load_summary(path2)

    name1 = "Wan2.1-T2V-Baseline"
    name2 = "Wan2.1-T2V-CoT"

    out_dir = Path("/home/siweim/code/PhyMetric/output100/compare_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overall_metrics(sum1, sum2, name1, name2, out_dir)
    plot_per_dimension(sum1, sum2, name1, name2, out_dir)

    print("done.")


if __name__ == "__main__":
    main()