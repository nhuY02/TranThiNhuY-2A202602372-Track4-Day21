"""CP4 — tạo ảnh failure case (side-by-side baseline vs yaw 2° drift trên frame 000049).

    python -m src.make_failure_figure
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

from starter.datasets import load_frame
from starter.projection import perturb_extrinsic, project_velo_to_image, overlay_points, draw_box2d


def main() -> None:
    fr = load_frame("data/kitti_mini", "000049")
    calib_ok = fr["calib"]
    calib_bad = perturb_extrinsic(calib_ok, yaw_deg=2.0)
    shape = fr["image"].shape

    uv_ok, depth_ok, _ = project_velo_to_image(fr["points"], calib_ok, shape)
    uv_bad, depth_bad, _ = project_velo_to_image(fr["points"], calib_bad, shape)

    vis_ok = overlay_points(fr["image"], uv_ok, depth_ok)
    vis_bad = overlay_points(fr["image"], uv_bad, depth_bad)
    for obj in fr["labels"]:
        vis_ok = draw_box2d(vis_ok, obj.bbox, color=(0, 255, 0), label=obj.type)
        vis_bad = draw_box2d(vis_bad, obj.bbox, color=(0, 255, 0), label=obj.type)

    # annotate
    def banner(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (img.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out

    vis_ok = banner(vis_ok, "Baseline (yaw=0deg)")
    vis_bad = banner(vis_bad, "Yaw drift +2deg  <- FAILURE")

    # side-by-side
    combined = np.hstack([vis_ok, vis_bad])
    out = Path("results/figures/fail_01_yaw2deg_frame000049.png")
    cv2.imwrite(str(out), combined)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
