"""Kiểm tra nhanh 2 hàm TODO(CP2) trong starter/projection.py.

    python -m src.test_projection

Kỳ vọng (CHECKPOINTS.md, CP2): với calib data/synthetic frame 000000, điểm velodyne (10, 0, 0)
có z_cam ~ 9.73 và pixel (u, v) ~ (614, 175); điểm NaN/Inf và điểm sau camera bị loại, không crash.
"""
from __future__ import annotations

import numpy as np

from starter.kitti_io import frame_paths, load_calib
from starter.projection import cam_to_image, velo_to_cam


def main() -> None:
    calib = load_calib(frame_paths("data/synthetic", "000000")["calib"])
    pts = np.array([
        [10.0, 0.0, 0.0],        # trước xe 10 m -> phải hợp lệ
        [-10.0, 0.0, 0.0],       # sau xe -> z_cam < 0, phải bị loại
        [np.nan, 0.0, 0.0],      # NaN -> phải bị loại
        [np.inf, 1.0, 0.0],      # Inf -> phải bị loại
        [10.0, 100.0, 0.0],      # lệch trái rất xa -> ngoài khung ảnh
    ])
    cam = velo_to_cam(pts, calib)
    uv, depth, mask = cam_to_image(cam, calib.P2, (375, 1242, 3))
    print(f"z_cam(10,0,0) = {cam[0, 2]:.3f}")
    print(f"uv(10,0,0)    = ({uv[0, 0]:.1f}, {uv[0, 1]:.1f}), depth = {depth[0]:.3f}")
    print(f"mask          = {mask.tolist()}")
    assert abs(cam[0, 2] - 9.73) < 0.05, "z_cam sai"
    assert abs(uv[0, 0] - 614) < 2 and abs(uv[0, 1] - 175) < 2, "pixel sai"
    assert mask.tolist() == [True, False, False, False, False], "lọc điểm sai"
    print("[PASS] velo_to_cam + cam_to_image")


if __name__ == "__main__":
    main()
