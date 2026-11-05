"""Export standalone figures from actual experimental results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_plots(results: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid")
    for metric in ("hit_rate_pct", "average_response_ms", "backend_accesses"):
        plot = sns.relplot(data=results, x="capacity_bytes", y=metric, hue="policy",
                           col="workload", kind="line", marker="o", errorbar="sd")
        plot.set_axis_labels("Cache capacity (bytes)", metric.replace("_", " "))
        plot.figure.savefig(output / f"{metric}.png", dpi=160, bbox_inches="tight")
        plt.close(plot.figure)
