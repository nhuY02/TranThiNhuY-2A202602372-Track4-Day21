"""Topic A — đo độ nhạy của LiDAR-camera projection với calibration drift (CP3).

Ý tưởng metric chính ("% điểm rơi đúng vào 2D box"):
  1. Dùng calibration GỐC để lấy các điểm LiDAR nằm trong 3D box GT của từng object
     (kiểm tra trong rectified camera frame, theo quy ước KITTI: location = tâm đáy, dims = h, w, l).
     Tập điểm này là "điểm thuộc object", không phụ thuộc vào mức perturb.
  2. Chiếu lại đúng các điểm đó bằng calibration ĐÃ PERTURB (projection.perturb_extrinsic).
  3. in_box = tỉ lệ điểm của object rơi vào 2D box của chính object đó (điểm bị loại do sau camera /
     ngoài ảnh tính là trượt). Trung bình theo object (mỗi object một phiếu), chia theo khoảng cách.

Mỗi lần chỉ thay đổi MỘT trục (yaw, pitch, roll, tx, ty, tz); frame, danh sách object, ngưỡng giữ nguyên.
Không có phép ngẫu nhiên nào, chạy lại cho ra đúng cùng số (seed vẫn được cố định cho chắc chắn).

    python -m src.calib_qa --help
    python -m src.calib_qa                              # cả 2 dataset, cấu hình mặc định
    python -m src.calib_qa --datasets data/kitti_mini --axes yaw tx
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from starter.datasets import dataset_type, list_frames, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam

SEED = 0
# Các mức perturb: góc tính bằng độ, tịnh tiến tính bằng mét (trong velodyne frame).
DEFAULT_LEVELS = {
    "yaw": [0.0, 0.5, 1.0, 2.0, 3.0],
    "pitch": [0.0, 0.5, 1.0, 2.0, 3.0],
    "roll": [0.0, 0.5, 1.0, 2.0, 3.0],
    "tx": [0.0, 0.02, 0.05, 0.10],
    "ty": [0.0, 0.02, 0.05, 0.10],
    "tz": [0.0, 0.02, 0.05, 0.10],
}
DIST_BINS = [(0.0, 15.0, "near"), (15.0, 30.0, "mid"), (30.0, np.inf, "far")]
MIN_OBJ_POINTS = 10


def default_frames(data_root: str) -> list[str]:
    """kitti: cả 20 frame. nuScenes: mỗi scene lấy 1/4 keyframe (0, 4, ..., 36) -> 20 frame."""
    frames = list_frames(data_root)
    if dataset_type(data_root) == "nuscenes":
        return [f for f in frames if int(f.rsplit("_", 1)[1]) % 4 == 0]
    return frames


def apply_drift(calib: KittiCalib, axis: str, level: float) -> KittiCalib:
    if axis in ("roll", "pitch", "yaw"):
        return perturb_extrinsic(calib, **{f"{axis}_deg": level})
    t = {"tx": (level, 0, 0), "ty": (0, level, 0), "tz": (0, 0, level)}[axis]
    return perturb_extrinsic(calib, t_xyz_m=t)


def points_in_box3d(pts_cam: np.ndarray, obj: KittiObject) -> np.ndarray:
    """Mask (N,) các điểm (rectified camera frame) nằm trong 3D box KITTI của obj."""
    h, w, l = obj.dimensions
    d = pts_cam - obj.location                      # gốc tại tâm đáy box
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    # Xoay ngược quanh trục y camera: R(ry)^T · d
    x_loc = c * d[:, 0] - s * d[:, 2]
    z_loc = s * d[:, 0] + c * d[:, 2]
    y_loc = d[:, 1]
    return (np.abs(x_loc) <= l / 2) & (np.abs(z_loc) <= w / 2) & (y_loc <= 0) & (y_loc >= -h)


@dataclass
class ObjPoints:
    idx: int
    obj: KittiObject
    dist_m: float
    pts_velo: np.ndarray       # (K, 3) điểm LiDAR thuộc object (lấy bằng calib gốc)
    uv_base: np.ndarray        # (K, 2) pixel khi chiếu bằng calib gốc (NaN nếu không chiếu được)


def project_all(pts_velo: np.ndarray, calib: KittiCalib, image_shape) -> np.ndarray:
    """Chiếu và trả (K, 2) pixel, NaN với điểm không hợp lệ (sau camera / ngoài ảnh / NaN)."""
    uv, _, mask = cam_to_image(velo_to_cam(pts_velo, calib), calib.P2, image_shape)
    out = np.full((len(pts_velo), 2), np.nan)
    out[mask] = uv
    return out


def extract_objects(fr: dict) -> list[ObjPoints]:
    pts = fr["points"][:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    cam = velo_to_cam(pts, fr["calib"])
    objs = []
    for i, obj in enumerate(fr["labels"]):
        x1, y1, x2, y2 = obj.bbox
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        m = points_in_box3d(cam, obj)
        if m.sum() < MIN_OBJ_POINTS:
            continue
        p = pts[m]
        objs.append(ObjPoints(i, obj, float(np.hypot(obj.location[0], obj.location[2])), p,
                              project_all(p, fr["calib"], fr["image"].shape)))
    return objs


def in_box_ratio(uv: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = bbox
    ok = np.isfinite(uv).all(axis=1)
    inside = ok.copy()
    inside[ok] = (uv[ok, 0] >= x1) & (uv[ok, 0] <= x2) & (uv[ok, 1] >= y1) & (uv[ok, 1] <= y2)
    return float(inside.mean())


def dist_bin(d: float) -> str:
    return next(name for lo, hi, name in DIST_BINS if lo <= d < hi)


def run_sweep(data_root: str, frames: list[str], levels: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = Path(data_root).name
    obj_rows, frame_rows = [], []
    for fid in frames:
        fr = load_frame(data_root, fid)
        calib, shape = fr["calib"], fr["image"].shape
        pts = fr["points"][:, :3]
        n_fov_base = int(cam_to_image(velo_to_cam(pts, calib), calib.P2, shape)[2].sum())
        objs = extract_objects(fr)
        for axis, lv in levels.items():
            for level in lv:
                c = apply_drift(calib, axis, level)
                n_fov = int(cam_to_image(velo_to_cam(pts, c), c.P2, shape)[2].sum())
                frame_rows.append(dict(dataset=ds, frame=fid, axis=axis, level=level,
                                       n_points=len(pts), n_in_fov=n_fov,
                                       pct_in_fov=n_fov / len(pts), fov_ratio_vs_base=n_fov / max(n_fov_base, 1)))
                for o in objs:
                    uv = project_all(o.pts_velo, c, shape)
                    shift = np.linalg.norm(uv - o.uv_base, axis=1)
                    obj_rows.append(dict(dataset=ds, frame=fid, obj_idx=o.idx, type=o.obj.type,
                                         occluded=o.obj.occluded, truncated=round(o.obj.truncated, 3),
                                         dist_m=round(o.dist_m, 2), dist_bin=dist_bin(o.dist_m),
                                         bbox_w_px=round(o.obj.bbox[2] - o.obj.bbox[0], 1),
                                         n_obj_points=len(o.pts_velo), axis=axis, level=level,
                                         in_box=in_box_ratio(uv, o.obj.bbox),
                                         px_shift_median=float(np.nanmedian(shift)) if np.isfinite(shift).any() else np.nan))
        print(f"  {ds}/{fid}: {len(objs)} objects with >= {MIN_OBJ_POINTS} points")
    return pd.DataFrame(obj_rows), pd.DataFrame(frame_rows)


def summarize(obj_df: pd.DataFrame, frame_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "axis", "level"]
    g = obj_df.groupby(keys)
    summ = g.agg(n_objects=("in_box", "size"), in_box_all=("in_box", "mean"),
                 px_shift_median=("px_shift_median", "median")).reset_index()
    for _, _, name in DIST_BINS:
        sub = obj_df[obj_df.dist_bin == name].groupby(keys)["in_box"]
        summ = summ.merge(sub.mean().rename(f"in_box_{name}").reset_index(), on=keys, how="left")
        summ = summ.merge(sub.size().rename(f"n_{name}").reset_index(), on=keys, how="left")
    fov = frame_df.groupby(keys).agg(n_frames=("frame", "nunique"), pct_in_fov=("pct_in_fov", "mean"),
                                     fov_ratio_vs_base=("fov_ratio_vs_base", "mean")).reset_index()
    summ = fov.merge(summ, on=keys)
    # Giá trị tương đối so với mức 0 của cùng dataset/trục: phần điểm "mất" do drift
    base = summ[summ.level == 0].set_index(["dataset", "axis"])["in_box_all"]
    summ["in_box_rel_to_base"] = summ.apply(lambda r: r.in_box_all / base[(r.dataset, r.axis)], axis=1)
    order = {a: i for i, a in enumerate(DEFAULT_LEVELS)}
    summ = summ.sort_values(["dataset", "axis", "level"], key=lambda s: s.map(order) if s.name == "axis" else s)
    return summ.round(4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep calibration drift (một trục mỗi lần) -> % điểm object rơi đúng 2D box")
    ap.add_argument("--datasets", nargs="+", default=["data/kitti_mini", "data/nuscenes_mini_subset"])
    ap.add_argument("--axes", nargs="+", default=list(DEFAULT_LEVELS), choices=list(DEFAULT_LEVELS))
    ap.add_argument("--frames", nargs="*", default=None, help="mặc định: xem default_frames()")
    ap.add_argument("--out-prefix", default="results/calib_perturb", help="tiền tố file CSV đầu ra")
    args = ap.parse_args()
    np.random.seed(SEED)

    levels = {a: DEFAULT_LEVELS[a] for a in args.axes}
    t0 = time.perf_counter()
    obj_all, frame_all = [], []
    for root in args.datasets:
        frames = args.frames or default_frames(root)
        print(f"[{root}] {len(frames)} frame, axes: {list(levels)}")
        o, f = run_sweep(root, frames, levels)
        obj_all.append(o)
        frame_all.append(f)
    obj_df, frame_df = pd.concat(obj_all), pd.concat(frame_all)
    summ = summarize(obj_df, frame_df)

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    obj_df.round(4).to_csv(f"{out}_per_object.csv", index=False)
    summ.to_csv(f"{out}_sweep.csv", index=False)
    cols = ["dataset", "axis", "level", "n_objects", "pct_in_fov", "in_box_all", "in_box_near", "in_box_mid",
            "in_box_far", "px_shift_median"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summ[cols].to_string(index=False))
    print(f"-> {out}_sweep.csv, {out}_per_object.csv  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
