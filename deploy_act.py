"""
ACT Policy Deployment — Bimanual SO-100  (with trajectory recording)
====================================================================
Runs the pretrained ACT model on two SO-100 follower arms with two cameras
and records every tick to disk so the run can be compared against the
LeRobot training dataset.

Recording layout (when --record-dir is set):
    <record-dir>/
        episode_000.parquet      # one row per tick
        episode_000_cam0.mp4     # 320x240 RGB, args.fps
        episode_000_cam1.mp4
        meta.json                # run config + joint key order

Each parquet row contains:
    timestamp, step_idx, chunk_remaining,
    observation.state           (12,)  float32  — concatenated [arm1, arm2] in degrees
    action.raw                  (12,)  float32  — direct policy output (post-postprocessor)
    action.applied              (12,)  float32  — what was actually sent to the arms
    gripper1_raw, gripper2_raw  scalars (raw policy gripper output before hysteresis)

This schema is a strict superset of the LeRobot dataset's
{observation.state, action} so you can load both into a DataFrame and diff.

Usage:
    python deploy_act.py --model ./pretrained_model --cam0 0 --cam1 1 \\
        --record-dir ./runs/$(date +%Y%m%d_%H%M%S)

Controls:
    Ctrl-C   stop, finalize recording, release torque
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from safetensors.torch import load_file

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy ACT policy on bimanual SO-100")
    parser.add_argument("--model",      default="./pretrained_model")
    parser.add_argument("--port1",      default="/dev/tty.usbmodem5AE60581371")
    parser.add_argument("--port2",      default="/dev/tty.usbmodem5AE60846081")
    parser.add_argument("--cam0",       type=int, default=0)
    parser.add_argument("--cam1",       type=int, default=1)
    parser.add_argument("--fps",        type=int, default=20)
    parser.add_argument("--step-scale", type=float, default=0.5,
                        help="Fraction of delta to apply per tick (default 0.5)")
    parser.add_argument("--max-step",   type=float, default=2.0,
                        help="Max degrees per joint per tick (default 2.0)")
    parser.add_argument("--show-cameras", action="store_true")
    parser.add_argument("--calibration-dir", type=Path, default=Path("."))

    # Recording
    parser.add_argument("--record-dir", type=Path, default=None,
                        help="If set, record trajectory + camera video into this dir.")
    parser.add_argument("--episode-idx", type=int, default=0,
                        help="Episode index used in output filenames.")
    parser.add_argument("--no-record-video", action="store_true",
                        help="Disable MP4 recording; still write parquet.")
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


# ── device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── model loading ─────────────────────────────────────────────────────────────

def load_act_policy(model_path: str, device: str):
    model_path = str(Path(model_path).resolve())
    print(f"Loading ACT policy from: {model_path}  device={device}")

    cfg = PreTrainedConfig.from_pretrained(model_path)
    cfg.pretrained_path = model_path
    cfg.device = device

    PolicyCls = get_policy_class("act")
    policy = PolicyCls(cfg)
    state_dict = load_file(str(Path(model_path) / "model.safetensors"))
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=model_path,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    print("ACT policy loaded.")
    return policy, preprocessor, postprocessor


# ── camera helpers ────────────────────────────────────────────────────────────

CAM_H, CAM_W = 240, 320


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    print(f"  Warming up camera {index}...", end="", flush=True)
    for _ in range(100):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError(f"Camera {index} failed to deliver frames after warmup")
    print(" ready")
    return cap


def read_camera_tensor(cap: cv2.VideoCapture) -> torch.Tensor:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(frame).permute(2, 0, 1).float()


# ── gripper hysteresis ────────────────────────────────────────────────────────

GRIPPER_OPEN_THRESHOLD  = 3.5
GRIPPER_CLOSE_THRESHOLD = 2.3
GRIPPER_OPEN_POS        = 4.5
GRIPPER_CLOSE_POS       = 2.0


def apply_gripper_hysteresis(action_val: float, current_val: float) -> float:
    if action_val > GRIPPER_OPEN_THRESHOLD:
        return GRIPPER_OPEN_POS
    elif action_val < GRIPPER_CLOSE_THRESHOLD:
        return GRIPPER_CLOSE_POS
    else:
        return current_val


# ── incremental target builder ────────────────────────────────────────────────

def build_target(joint_keys, current, action, step_scale, max_step):
    delta = action - current
    step  = np.clip(delta * step_scale, -max_step, max_step)
    target = {}
    for i, k in enumerate(joint_keys):
        if "gripper" in k:
            target[k] = apply_gripper_hysteresis(action[i], current[i])
        else:
            target[k] = float(current[i] + step[i])
    return target


# ── trajectory recorder ───────────────────────────────────────────────────────

class TrajectoryRecorder:
    """
    Buffers per-tick rows in memory and flushes to a parquet file on close().
    Also writes one MP4 per camera if record_video=True.

    The schema mirrors LeRobot's dataset columns where it overlaps:
        observation.state, action  (both as fixed-length float32 lists)
    plus extra deployment-only fields for diagnostics.
    """

    def __init__(self, out_dir: Path, episode_idx: int, fps: int,
                 joint_keys1, joint_keys2, record_video: bool):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = episode_idx
        self.fps = fps
        self.joint_keys = list(joint_keys1) + list(joint_keys2)
        self.record_video = record_video

        self.rows = []
        self.t_start = None

        self._writers = {}
        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            for cam in ("cam0", "cam1"):
                path = self.out_dir / f"episode_{episode_idx:03d}_{cam}.mp4"
                self._writers[cam] = cv2.VideoWriter(
                    str(path), fourcc, fps, (CAM_W, CAM_H))

        meta = {
            "episode_idx": episode_idx,
            "fps": fps,
            "joint_keys": self.joint_keys,
            "schema": {
                "observation.state": f"float32[{len(self.joint_keys)}]",
                "action.raw":        f"float32[{len(self.joint_keys)}]",
                "action.applied":    f"float32[{len(self.joint_keys)}]",
            },
            "notes": (
                "action.raw is the direct policy output AFTER postprocessing. "
                "action.applied is what was sent to the arms (after delta-stepping "
                "and gripper hysteresis). Compare action.raw against the training "
                "dataset's `action` column; compare observation.state against the "
                "training dataset's `observation.state`."
            ),
        }
        with open(self.out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def log(self, *, step_idx, state, action_raw, action_applied,
            gripper1_raw, gripper2_raw, chunk_remaining,
            cam0_frame=None, cam1_frame=None):
        if self.t_start is None:
            self.t_start = time.time()
        ts = time.time() - self.t_start

        self.rows.append({
            "timestamp":         float(ts),
            "step_idx":          int(step_idx),
            "chunk_remaining":   int(chunk_remaining),
            "observation.state": np.asarray(state, dtype=np.float32).tolist(),
            "action.raw":        np.asarray(action_raw, dtype=np.float32).tolist(),
            "action.applied":    np.asarray(action_applied, dtype=np.float32).tolist(),
            "gripper1_raw":      float(gripper1_raw),
            "gripper2_raw":      float(gripper2_raw),
        })

        if self.record_video:
            if cam0_frame is not None and "cam0" in self._writers:
                self._writers["cam0"].write(
                    cv2.cvtColor(cam0_frame, cv2.COLOR_RGB2BGR))
            if cam1_frame is not None and "cam1" in self._writers:
                self._writers["cam1"].write(
                    cv2.cvtColor(cam1_frame, cv2.COLOR_RGB2BGR))

    def close(self):
        for w in self._writers.values():
            w.release()
        self._writers.clear()

        if not self.rows:
            print("Recorder: no rows to save.")
            return

        out_path = self.out_dir / f"episode_{self.episode_idx:03d}.parquet"
        try:
            import pandas as pd
            df = pd.DataFrame(self.rows)
            df.to_parquet(out_path, index=False)
            print(f"Recorder: wrote {len(df)} rows -> {out_path}")
        except Exception as e:
            # Fallback to npz if pandas/pyarrow unavailable
            fallback = self.out_dir / f"episode_{self.episode_idx:03d}.npz"
            print(f"Recorder: parquet write failed ({e}); saving npz instead.")
            np.savez(
                fallback,
                timestamp=np.array([r["timestamp"] for r in self.rows], dtype=np.float32),
                step_idx=np.array([r["step_idx"] for r in self.rows], dtype=np.int64),
                state=np.stack([np.array(r["observation.state"], dtype=np.float32)
                                for r in self.rows]),
                action_raw=np.stack([np.array(r["action.raw"], dtype=np.float32)
                                     for r in self.rows]),
                action_applied=np.stack([np.array(r["action.applied"], dtype=np.float32)
                                         for r in self.rows]),
                gripper1_raw=np.array([r["gripper1_raw"] for r in self.rows], dtype=np.float32),
                gripper2_raw=np.array([r["gripper2_raw"] for r in self.rows], dtype=np.float32),
            )
            print(f"Recorder: wrote {len(self.rows)} rows -> {fallback}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    policy, preprocessor, postprocessor = load_act_policy(args.model, device)

    robot1 = SO100Follower(SO100FollowerConfig(
        port=args.port1, id="arm1", use_degrees=True,
        calibration_dir=args.calibration_dir))
    robot2 = SO100Follower(SO100FollowerConfig(
        port=args.port2, id="arm2", use_degrees=True,
        calibration_dir=args.calibration_dir))

    print("Connecting robots...")
    robot1.connect()
    robot2.connect()
    print("Connected.")

    obs1 = robot1.get_observation()
    obs2 = robot2.get_observation()
    joint_keys1 = [k for k in obs1 if k.endswith(".pos")]
    joint_keys2 = [k for k in obs2 if k.endswith(".pos")]
    print("arm1 joints:", joint_keys1)
    print("arm2 joints:", joint_keys2)

    print(f"Opening cameras (cam0={args.cam0}, cam1={args.cam1})...")
    cap0 = open_camera(args.cam0)
    cap1 = open_camera(args.cam1)
    print(f"Cameras ready. step_scale={args.step_scale} max_step={args.max_step}")

    # Recorder
    recorder = None
    if args.record_dir is not None:
        recorder = TrajectoryRecorder(
            out_dir=args.record_dir,
            episode_idx=args.episode_idx,
            fps=args.fps,
            joint_keys1=joint_keys1,
            joint_keys2=joint_keys2,
            record_video=not args.no_record_video,
        )
        print(f"Recording to {args.record_dir} (episode {args.episode_idx:03d})")

    action_chunk = deque()
    chunk_size   = 10
    step_idx     = 0

    print(f"\nRunning at {args.fps} fps. Press Ctrl-C to stop.\n")

    try:
        while True:
            t0 = time.time()

            obs1 = robot1.get_observation()
            obs2 = robot2.get_observation()
            current1 = np.array([obs1[k] for k in joint_keys1], dtype=np.float32)
            current2 = np.array([obs2[k] for k in joint_keys2], dtype=np.float32)
            state = np.concatenate([current1, current2])

            cam0_t = read_camera_tensor(cap0)
            cam1_t = read_camera_tensor(cap1)

            # Keep a uint8 RGB copy for video recording / preview
            cam0_rgb = cam0_t.permute(1, 2, 0).numpy().astype(np.uint8)
            cam1_rgb = cam1_t.permute(1, 2, 0).numpy().astype(np.uint8)

            if args.show_cameras:
                cv2.imshow(
                    "cam0 | cam1",
                    np.hstack([
                        cv2.cvtColor(cam0_rgb, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(cam1_rgb, cv2.COLOR_RGB2BGR),
                    ]),
                )
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            if not action_chunk:
                obs = {
                    "observation.images.cam0": cam0_t,
                    "observation.images.cam1": cam1_t,
                    "observation.state": torch.from_numpy(state),
                }
                with torch.no_grad():
                    policy_input  = preprocessor(obs)
                    policy_output = policy.select_action(policy_input)
                    actions_raw   = postprocessor(policy_output)

                if isinstance(actions_raw, torch.Tensor):
                    actions_raw = actions_raw.detach().cpu().numpy()
                actions_raw = np.asarray(actions_raw)

                if actions_raw.ndim == 3:
                    actions_raw = actions_raw[0]
                elif actions_raw.ndim == 1:
                    actions_raw = actions_raw[np.newaxis]

                for a in actions_raw[:chunk_size]:
                    action_chunk.append(a)

            action  = action_chunk.popleft()
            action1 = action[:6]
            action2 = action[6:]

            target1 = build_target(joint_keys1, current1, action1,
                                   args.step_scale, args.max_step)
            target2 = build_target(joint_keys2, current2, action2,
                                   args.step_scale, args.max_step)

            robot1.send_action(target1)
            robot2.send_action(target2)

            # Build the "applied" vector in the same joint order as state
            applied1 = np.array([target1[k] for k in joint_keys1], dtype=np.float32)
            applied2 = np.array([target2[k] for k in joint_keys2], dtype=np.float32)
            action_applied = np.concatenate([applied1, applied2])

            if recorder is not None:
                recorder.log(
                    step_idx=step_idx,
                    state=state,
                    action_raw=np.asarray(action, dtype=np.float32),
                    action_applied=action_applied,
                    gripper1_raw=float(action1[-1]),
                    gripper2_raw=float(action2[-1]),
                    chunk_remaining=len(action_chunk),
                    cam0_frame=cam0_rgb,
                    cam1_frame=cam1_rgb,
                )

            if step_idx % args.print_every == 0:
                print(f"[step {step_idx:05d}]  chunk_remaining={len(action_chunk)}")
                print(f"  arm1  cur: {np.round(current1, 2)}")
                print(f"  arm1  raw: {np.round(action1,  2)}")
                print(f"  arm1  app: {np.round(applied1, 2)}")
                print(f"  arm2  cur: {np.round(current2, 2)}")
                print(f"  arm2  raw: {np.round(action2,  2)}")
                print(f"  arm2  app: {np.round(applied2, 2)}")
                print(f"  gripper1 raw: {action1[-1]:.3f}  cmd: {target1['gripper.pos']:.2f}")
                print(f"  gripper2 raw: {action2[-1]:.3f}  cmd: {target2['gripper.pos']:.2f}")

            step_idx += 1

            dt = time.time() - t0
            time.sleep(max(1.0 / args.fps - dt, 0))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("Releasing torque and disconnecting...")
        if recorder is not None:
            recorder.close()
        cap0.release()
        cap1.release()
        if args.show_cameras:
            cv2.destroyAllWindows()
        try:
            robot1.disconnect()
            robot2.disconnect()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()