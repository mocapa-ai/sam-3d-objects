#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Run SAM-3D-Objects inference and log mesh pose to Rerun in world frame.

Frame notes:
- `output["translation"]` and `output["rotation"]` come from
  `instance_position_l2c` / `instance_quaternion_l2c`, i.e. object-local -> camera.
- In this pipeline that camera frame is PyTorch3D camera coordinates.
- hand_object_tracking extrinsics (`camFrame_labFrame`) use OpenCV camera coordinates.
- We convert PyTorch3D -> OpenCV with S = diag(-1, -1, 1), then compose with camera
  extrinsics to obtain object pose in world/lab frame.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple
from uuid import uuid4

import numpy as np
import rerun as rr
import cv2
from scipy.spatial.transform import Rotation as SciRot


# demo.py-style import path
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log SAM-3D-Objects inference to Rerun.")
    parser.add_argument("--config-path", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument("--image-path", required=True, help="Input RGB/RGBA image path.")
    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Directory containing binary mask images. See load_single_mask().",
    )
    parser.add_argument("--mask-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-model", action="store_true", default=False)
    parser.add_argument("--with-layout-postprocess", action="store_true", default=False)

    parser.add_argument(
        "--markerless-data-dir",
        default=str(
            Path(__file__).resolve().parent.parent
            / "hand_object_tracking"
            / "data"
            / "markerless_data"
        ),
        help="Path to markerless_data containing recordings/ and intrinsics/.",
    )
    parser.add_argument(
        "--recording-name",
        required=True,
        help="Recording short name used in '*_<recording-name>' under recordings/.",
    )
    parser.add_argument("--camera-id", default='234500078', help="Camera id for intrinsics/extrinsics.")

    parser.add_argument("--recording-id", default=None, help="Existing Rerun recording id.")
    parser.add_argument("--connect-grpc", action="store_true", default=False)
    parser.add_argument("--save-rrd", default=None, help="Optional .rrd output path.")
    parser.add_argument("--viz-scale", type=float, default=1.0, help="Display-only scale for translations.")
    parser.add_argument("--entity-path", default="world/sam3d_object")
    parser.add_argument("--export-glb-path", default=None, help="Optional GLB export path.")
    parser.add_argument("--log-camera", dest="log_camera", action="store_true", default=True)
    parser.add_argument("--no-log-camera", dest="log_camera", action="store_false")
    parser.add_argument(
        "--debug-pose-variants",
        action="store_true",
        default=False,
        help="Log l2c_p3d, l2c_cv, and l2w debug variants as separate entities.",
    )
    parser.add_argument(
        "--debug-spacing",
        type=float,
        default=0.30,
        help="Offset spacing (meters) used to separate debug entities visually.",
    )
    return parser.parse_args()


def find_recording_folder(markerless_data_dir: str, recording_name: str) -> str:
    matches = glob.glob(os.path.join(markerless_data_dir, "recordings", f"*_{recording_name}"))
    if not matches:
        raise FileNotFoundError(
            f"No recording folder matching '*_{recording_name}' in "
            f"{os.path.join(markerless_data_dir, 'recordings')}"
        )
    return matches[0]


def load_intrinsics_with_sources(intrinsics_dir: str) -> Tuple[Dict[str, dict], Dict[str, str]]:
    intrinsics: Dict[str, dict] = {}
    source_paths: Dict[str, str] = {}
    candidates = glob.glob(os.path.join(intrinsics_dir, "*_*", "*.pkl"))
    candidates.sort(reverse=True)
    for pkl_path in candidates:
        cam_id = os.path.basename(pkl_path).replace(".pkl", "")
        if cam_id in intrinsics:
            continue
        with open(pkl_path, "rb") as f:
            intrinsics[cam_id] = pickle.load(f)
        source_paths[cam_id] = pkl_path
    return intrinsics, source_paths


def load_extrinsics(recording_folder: str) -> Dict[str, dict]:
    extrinsics_path = os.path.join(recording_folder, "systemCalibration.pkl")
    with open(extrinsics_path, "rb") as f:
        return pickle.load(f)


def pose_matrix_from_t_q(t_xyz: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    t = np.asarray(t_xyz, dtype=np.float64).reshape(3)
    r = SciRot.from_quat(np.asarray(q_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def pose_t_q_from_matrix(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(T[:3, 3], dtype=np.float64).reshape(3)
    q = SciRot.from_matrix(np.asarray(T[:3, :3], dtype=np.float64)).as_quat()
    return t, q


def pytorch3d_to_opencv_pose(t_p3d: np.ndarray, q_p3d_xyzw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
    # PyTorch3D Transform3d applies row-vector transforms; convert to a column-vector
    # rotation before composition with OpenCV-style extrinsics.
    r_p3d_row = SciRot.from_quat(np.asarray(q_p3d_xyzw, dtype=np.float64)).as_matrix()
    r_p3d_col = r_p3d_row.T
    r_cv_col = s @ r_p3d_col @ s
    t_cv = s @ np.asarray(t_p3d, dtype=np.float64).reshape(3)
    q_cv_xyzw = SciRot.from_matrix(r_cv_col).as_quat()
    return t_cv, q_cv_xyzw


def compose_local_to_world(
    t_obj_cam_cv: np.ndarray,
    q_obj_cam_cv_xyzw: np.ndarray,
    t_wc: np.ndarray,
    r_wc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    T_obj_cam = pose_matrix_from_t_q(t_obj_cam_cv, q_obj_cam_cv_xyzw)
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = np.asarray(r_wc, dtype=np.float64)
    T_wc[:3, 3] = np.asarray(t_wc, dtype=np.float64).reshape(3)
    T_obj_world = T_wc @ T_obj_cam
    return pose_t_q_from_matrix(T_obj_world)


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def main() -> None:
    args = parse_args()

    if args.recording_id is None:
        args.recording_id = str(uuid4())

    recording_folder = find_recording_folder(args.markerless_data_dir, args.recording_name)
    intrinsics_dir = os.path.join(args.markerless_data_dir, "intrinsics")
    intrinsics, intrinsic_sources = load_intrinsics_with_sources(intrinsics_dir)
    extrinsics = load_extrinsics(recording_folder)

    if args.camera_id not in intrinsics:
        raise KeyError(f"Camera '{args.camera_id}' not found in intrinsics.")
    if args.camera_id not in extrinsics:
        raise KeyError(f"Camera '{args.camera_id}' not found in extrinsics.")

    print(f"[INFO] using camera_id={args.camera_id}")
    print(f"[INFO] intrinsics source: {intrinsic_sources[args.camera_id]}")

    inference = Inference(args.config_path, compile=args.compile_model)
    image = load_image(args.image_path)
    mask = load_single_mask(args.mask_dir, index=args.mask_index)
    output = inference(
        image,
        mask,
        seed=args.seed,
        # with_layout_postprocess=args.with_layout_postprocess,
    )

    if "glb" not in output or output["glb"] is None:
        raise RuntimeError("Inference output does not contain 'glb'.")
    if "translation" not in output or "rotation" not in output:
        raise RuntimeError("Inference output does not contain 'translation'/'rotation'.")

    glb = output["glb"]
    if args.export_glb_path is not None:
        glb.export(args.export_glb_path)
        glb_path = args.export_glb_path
    else:
        glb_path = os.path.abspath("mesh.glb")
        glb.export(glb_path)

    output_translation = output["translation"].detach().cpu().numpy()
    output_rotation = output["rotation"].detach().cpu().numpy()
    output_6d_rotation = output["rotation_6d"].detach().cpu().numpy() if "rotation_6d" in output else None

    print(f"[DEBUG] Raw output translation (l2c): {output_translation}")
    print(f"[DEBUG] Raw output rotation (l2c): {output_rotation}")
    print(f"[DEBUG] Raw output 6D rotation (l2c): {output_6d_rotation}")

    t_l2c_p3d = np.asarray(output_translation).reshape(-1, 3)[0].astype(np.float64)
    q_l2c_p3d = np.asarray(output_rotation).reshape(-1, 4)[0].astype(np.float64)
    q_l2c_p3d_xyzw = quat_wxyz_to_xyzw(q_l2c_p3d)

    print(f"[DEBUG] Parsed output translation (l2c, PyTorch3D cam): {t_l2c_p3d.tolist()}")
    print(f"[DEBUG] Parsed output rotation (l2c, PyTorch3D cam): {q_l2c_p3d_xyzw.tolist()}")

    t_l2c_cv, q_l2c_cv_xyzw = pytorch3d_to_opencv_pose(t_l2c_p3d, q_l2c_p3d_xyzw)

    print(f"[DEBUG] Converted output translation (l2c, OpenCV cam): {t_l2c_cv.tolist()}")
    print(f"[DEBUG] Converted output rotation (l2c, OpenCV cam): {q_l2c_cv_xyzw.tolist()}")

    t_cw = np.asarray(extrinsics[args.camera_id]["camFrame_labFrame"], dtype=np.float64)
    t_wc = np.linalg.inv(t_cw)
    r_wc = t_wc[:3, :3]
    twc = t_wc[:3, 3]
    t_l2w, q_l2w_xyzw = compose_local_to_world(t_l2c_cv, q_l2c_cv_xyzw, twc, r_wc)
    t_l2w_no_flip, q_l2w_no_flip_xyzw = compose_local_to_world(t_l2c_p3d, q_l2c_p3d_xyzw, twc, r_wc)

    s_l2c = output["scale"].detach().cpu().numpy().reshape(-1, 3)[0]

    stream = rr.RecordingStream("Hand Object Visualisation", recording_id=args.recording_id)
    if args.connect_grpc:
        stream.connect_grpc()
    if args.save_rrd is not None:
        stream.save(args.save_rrd)

    stream.set_time("frame_count", sequence=0)
    stream.log("world", rr.Transform3D(translation=[0.0, 0.0, 0.0]), static=True)

    if args.log_camera:
        k = np.asarray(intrinsics[args.camera_id]["K"], dtype=np.float64)
        camera_image_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        if camera_image_bgr is None:
            raise FileNotFoundError(f"Failed to read camera image at: {args.image_path}")
        img_h, img_w = camera_image_bgr.shape[:2]
        stream.log(
            f"cameras/camera_{args.camera_id}",
            rr.Pinhole(image_from_camera=k, resolution=[img_w, img_h]),
            static=True,
        )
        stream.log(
            f"cameras/camera_{args.camera_id}",
            rr.Transform3D(
                translation=(twc * args.viz_scale).tolist(),
                mat3x3=r_wc,
            ),
            static=True,
        )
        ok, jpeg = cv2.imencode(".jpg", camera_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"Failed to JPEG-encode image: {args.image_path}")
        stream.log(
            f"cameras/camera_{args.camera_id}/image",
            rr.EncodedImage(contents=jpeg.tobytes(), media_type="image/jpeg", opacity=1.0),
        )

    stream.log(f"world/camera_{args.camera_id}", rr.Transform3D(translation=twc.tolist(), mat3x3=r_wc), static=True)
    stream.log(f"world/camera_{args.camera_id}/mesh", rr.Asset3D(path=glb_path), static=True)
    stream.log(
        f"world/camera_{args.camera_id}/mesh",
        rr.Transform3D(
            translation=t_l2c_p3d , # (t_l2w * args.viz_scale).tolist(),
            quaternion=rr.Quaternion(xyzw=q_l2c_p3d_xyzw.tolist()),
            scale=s_l2c.tolist(),
        ),
    )
    def p3d_to_rerun(xyz):
        """Convert a (..., 3) tensor from PyTorch3D to Rerun/OpenCV space."""
        xyz = xyz.clone()
        xyz[..., 0] *= -1  # flip X
        xyz[..., 1] *= -1  # flip Y
        return xyz
    # Log pointmap as a point cloud
    pts = output["pointmap"].permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
    # pts_rerun = p3d_to_rerun(pts).numpy()
    colors = output["pointmap_colors"].permute(1, 2, 0).reshape(-1, 3).numpy()
    stream.log("scene/pointmap", rr.Points3D(pts, colors=colors))

    if args.debug_pose_variants:
        debug_spacing = float(args.debug_spacing) * args.viz_scale
        if args.log_camera:
            # Variant A: direct model output (l2c in PyTorch3D convention) under camera.
            stream.log(
                "cameras/debug_l2c_p3d/mesh",
                rr.Asset3D(path=glb_path),
                static=True,
            )
            stream.log(
                "cameras/debug_l2c_p3d",
                rr.Transform3D(
                    translation=(t_l2c_p3d * args.viz_scale + np.array([-debug_spacing, 0.0, 0.0])).tolist(),
                    quaternion=rr.Quaternion(xyzw=q_l2c_p3d_xyzw.tolist()),
                    scale=s_l2c.tolist(),
                ),
            )

            # Variant B: converted local->camera pose in OpenCV camera convention.
            stream.log(
                "cameras/debug_l2c_cv/mesh",
                rr.Asset3D(path=glb_path),
                static=True,
            )
            stream.log(
                "cameras/debug_l2c_cv",
                rr.Transform3D(
                    translation=(t_l2c_cv * args.viz_scale + np.array([debug_spacing, 0.0, 0.0])).tolist(),
                    quaternion=rr.Quaternion(xyzw=q_l2c_cv_xyzw.tolist()),
                    scale=s_l2c.tolist(),
                ),
            )
        else:
            print("[WARN] --debug-pose-variants requested but camera entities are disabled (--no-log-camera).")

        # Variant C: world pose after full composition.
        stream.log(
            "world/debug_l2w/mesh",
            rr.Asset3D(path=glb_path),
            static=True,
        )
        stream.log(
            "world/debug_l2w",
            rr.Transform3D(
                translation=(t_l2w * args.viz_scale + np.array([0.0, 0.0, debug_spacing])).tolist(),
                quaternion=rr.Quaternion(xyzw=q_l2w_xyzw.tolist()),
                scale=s_l2c.tolist(),
            ),
        )

        # Optional: no-flip world variant to quickly diagnose wrong-convention issues.
        stream.log(
            "world/debug_l2w_no_flip/mesh",
            rr.Asset3D(path=glb_path),
            static=True,
        )
        stream.log(
            "world/debug_l2w_no_flip",
            rr.Transform3D(
                translation=(t_l2w_no_flip * args.viz_scale + np.array([debug_spacing, 0.0, debug_spacing])).tolist(),
                quaternion=rr.Quaternion(xyzw=q_l2w_no_flip_xyzw.tolist()),
                scale=s_l2c.tolist(),
            ),
        )

    print("[INFO] logged mesh + world transform to rerun")
    print(f"[INFO] output translation (l2c, PyTorch3D cam): {t_l2c_p3d.tolist()}")
    print(f"[INFO] output rotation (l2c, wxyz, PyTorch3D cam): {q_l2c_p3d.tolist()}")
    print(f"[INFO] transformed translation (l2w, world): {t_l2w.tolist()}")
    print(f"[INFO] transformed rotation (l2w, xyzw): {q_l2w_xyzw.tolist()}")
    print(f"[INFO] recording_id: {args.recording_id}")


if __name__ == "__main__":
    main()
