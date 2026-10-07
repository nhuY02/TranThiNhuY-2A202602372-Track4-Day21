"""CP3 — vẽ biểu đồ kết quả calibration drift sweep (Topic A).

    python -m src.plot_results
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SWEEP = Path("results/calib_perturb_sweep.csv")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"near": "#e74c3c", "mid": "#e67e22", "far": "#2980b9", "all": "#2c3e50"}
DS_LABELS = {"kitti_mini": "KITTI (64-beam)", "nuscenes_mini_subset": "nuScenes (32-beam)"}


def plot_yaw_by_dist(df: pd.DataFrame) -> None:
    """% điểm rơi đúng 2D box theo mức yaw drift, chia theo khoảng cách."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (ds, sdf) in zip(axes, df.groupby("dataset", sort=False)):
        ydf = sdf[sdf.axis == "yaw"]
        for band, label, col in [("near", "near (<15 m)", COLORS["near"]),
                                   ("mid",  "mid (15–30 m)", COLORS["mid"]),
                                   ("far",  "far (>30 m)",   COLORS["far"]),
                                   ("all",  "overall",        COLORS["all"])]:
            col_name = f"in_box_{band}" if band != "all" else "in_box_all"
            ax.plot(ydf["level"], ydf[col_name] * 100, "o-", color=col, label=label, linewidth=2)

        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(DS_LABELS.get(ds, ds), fontsize=13, fontweight="bold")
        ax.set_xlabel("Yaw drift (°)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_xticks([0, 0.5, 1.0, 2.0, 3.0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("% LiDAR points inside GT 2D box", fontsize=11)
    fig.suptitle("Calibration Drift (Yaw): Impact on LiDAR-Camera Projection\nby Distance Range",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "yaw_drift_by_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)


def plot_px_shift(df: pd.DataFrame) -> None:
    """Median pixel shift theo mức perturb cho yaw/tx/ty, hai dataset cạnh nhau."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (ds, sdf) in zip(axes, df.groupby("dataset", sort=False)):
        for axis, ls in [("yaw", "o-"), ("tx", "s--"), ("ty", "^:")]:
            adf = sdf[sdf.axis == axis]
            label = f"yaw (°)" if axis == "yaw" else f"{axis} (m)"
            ax.plot(adf["level"], adf["px_shift_median"], ls, label=label, linewidth=2)
        ax.set_title(DS_LABELS.get(ds, ds), fontsize=13, fontweight="bold")
        ax.set_xlabel("Perturbation magnitude", fontsize=11)
        ax.set_ylabel("Median pixel shift (px)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Median Pixel Shift vs Perturbation Magnitude\n(yaw / tx / ty)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT / "px_shift_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)


def plot_both_datasets_yaw(df: pd.DataFrame) -> None:
    """% in_box_far so sánh 2 dataset trên cùng 1 plot — highlight khác biệt 64 vs 32 beam."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds, sdf in df.groupby("dataset", sort=False):
        ydf = sdf[sdf.axis == "yaw"]
        ax.plot(ydf["level"], ydf["in_box_far"] * 100, "o-",
                label=DS_LABELS.get(ds, ds), linewidth=2)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Yaw drift (°)", fontsize=12)
    ax.set_ylabel("% points (far >30 m) inside GT 2D box", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xticks([0, 0.5, 1.0, 2.0, 3.0])
    ax.set_title("Yaw Drift Effect on Far Objects\n(>30 m): KITTI 64-beam vs nuScenes 32-beam",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT / "yaw_far_both_datasets.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(SWEEP)
    plot_yaw_by_dist(df)
    plot_px_shift(df)
    plot_both_datasets_yaw(df)
    print("All plots done.")


if __name__ == "__main__":
    main()
