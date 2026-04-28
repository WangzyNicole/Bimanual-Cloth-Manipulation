"""
ACT Policy Deployment — Bimanual SO-100  (ticks I/O, radian model)
==================================================================
The ACT policy was trained on data converted via:
    state_rad = ticks * (pi / 2048)
(see convert_new_class_data.py + configs/train_act_class170.yaml).

So the model expects radians on its inputs and outputs radians on its action.
The servos, however, speak raw ticks (PRESENT/GOAL_POSITION).  This script:

    motors  --(ticks)-->  ticks*(pi/2048)  --(radians)-->  ACT  --(radians)-->
                                                                       |
                                                                       v
    motors  <--(ticks)--  action_rad*(2048/pi)  <--------------(radians)

Recording layout (when --record-dir is set):
    <record-dir>/
        episode_000.parquet    # one row per tick
        episode_000_cam0.mp4
        episode_000_cam1.mp4
        meta.json

Each parquet row contains (everything in radians for direct comparability
to the training dataset, plus raw tick columns for debugging):
    timestamp, step_idx, chunk_remaining,
    observation.state        (12,)  float32  radians
    observation.state_ticks  (12,)  float32  raw servo ticks
    action.raw               (12,)  float32  policy output in radians
    action.applied           (12,)  float32  applied target in radians
    action.applied_ticks     (12,)  float32  what was actually written to the motors
    gripper1_raw, gripper2_raw       (raw policy gripper output before hysteresis)

Usage:
    python deploy_act.py --model ./pretrained_model --cam0 0 --cam1 1 \\
        --record-dir ./runs/$(date +%Y%m%d_%H%M%S)

Controls:
    Ctrl-C   stop, finalize recording, release torque
"""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from safetensors.torch import load_file

from scservo_sdk import PortHandler, PacketHandler


# ── servo constants (FEETECH / SCS protocol) ──────────────────────────────────
TORQUE_ADDR        = 40
GOAL_POSITION_ADDR = 42
PRESENT_ADDR       = 56

TICKS_TO_RAD = math.pi / 2048.0     # must match convert_new_class_data.py
RAD_TO_TICKS = 2048.0 / math.pi
TICK_MIN, TICK_MAX = 0, 4095


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy ACT policy on bimanual SO-100 (ticks ↔ radians)")
    parser.add_argument("--model",      default="./pretrained_model")
    parser.add_argument("--port1",      default="/dev/tty.usbmodem5AE60581371",
                        help="Serial port for arm1 (RHS)")
    parser.add_argument("--port2",      default="/dev/tty.usbmodem5AE60846081",
                        help="Serial port for arm2 (LHS)")
    parser.add_argument("--cam0",       type=int, default=0)
    parser.add_argument("--cam1",       type=int, default=1)
    parser.add_argument("--fps",        type=int, default=20)
    parser.add_argument("--step-scale", type=float, default=0.5,
                        help="Fraction of (target-current) delta to apply per loop")
    parser.add_argument("--max-step",   type=int,   default=50,
                        help="Max ticks per joint per loop (default 50 ≈ 0.077 rad)")
    parser.add_argument("--show-cameras", action="store_true")

    # Recording
    parser.add_argument("--record-dir",   type=Path, default=None,
                        help="If set, record trajectory + camera video into this dir.")
    parser.add_argument("--episode-idx",  type=int, default=0)
    parser.add_argument("--no-record-video", action="store_true")
    parser.add_argument("--print-every",  type=int, default=20)
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


# ── motor I/O (raw ticks via scservo_sdk) ────────────────────────────────────

def open_port(name: str) -> PortHandler:
    p = PortHandler(name)
    if not p.openPort():
        raise RuntimeError(f"Cannot open port: {name}")
    p.setBaudRate(1_000_000)
    return p


def set_torque(ph: PacketHandler, port: PortHandler, on: bool):
    for sid in range(1, 7):
        ph.write1ByteTxRx(port, sid, TORQUE_ADDR, 1 if on else 0)


def read_ticks(ph: PacketHandler, port: PortHandler, last: np.ndarray) -> np.ndarray:
    """Read present positions for servo IDs 1..6.  On comm error, hold `last`."""
    out = np.empty(6, dtype=np.float32)
    for sid in range(1, 7):
        v, comm, _ = ph.read2ByteTxRx(port, sid, PRESENT_ADDR)
        out[sid - 1] = float(v) if comm == 0 else float(last[sid - 1])
    return out


def write_ticks(ph: PacketHandler, port: PortHandler, ticks: np.ndarray):
    """Write goal positions for servo IDs 1..6, clamped to [0, 4095]."""
    for sid in range(1, 7):
        t = int(np.clip(ticks[sid - 1], TICK_MIN, TICK_MAX))
        ph.write2ByteTxRx(port, sid, GOAL_POSITION_ADDR, t)


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


# ── gripper hysteresis (operates in RADIAN space — same as model output) ─────
# Gripper joint range in ticks is ~1425–2913, i.e. ~2.19–4.47 rad.
GRIPPER_OPEN_THRESH_RAD  = 3.5
GRIPPER_CLOSE_THRESH_RAD = 2.3
GRIPPER_OPEN_POS_RAD     = 4.5
GRIPPER_CLOSE_POS_RAD    = 2.0


def apply_gripper_hysteresis(action_rad: float, current_rad: float) -> float:
    if action_rad > GRIPPER_OPEN_THRESH_RAD:
        return GRIPPER_OPEN_POS_RAD
    if action_rad < GRIPPER_CLOSE_THRESH_RAD:
        return GRIPPER_CLOSE_POS_RAD
    return current_rad


# ── trajectory recorder ───────────────────────────────────────────────────────

class TrajectoryRecorder:
    """Buffers per-tick rows in memory and flushes to a parquet file on close()."""

    def __init__(self, out_dir: Path, episode_idx: int, fps: int,
                 joint_keys, record_video: bool):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episode_idx = episode_idx
        self.fps = fps
        self.joint_keys = list(joint_keys)
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
            "ticks_to_rad": TICKS_TO_RAD,
            "schema": {
                "observation.state":       "float32[12]  radians",
                "observation.state_ticks": "float32[12]  raw servo ticks",
                "action.raw":              "float32[12]  radians (policy output)",
                "action.applied":          "float32[12]  radians (commanded)",
                "action.applied_ticks":    "float32[12]  ticks  (written to motors)",
            },
            "notes": (
                "All radian columns are directly comparable to the training "
                "dataset's `observation.state` / `action` columns "
                "(ticks * pi/2048).  observation.state_ticks and "
                "action.applied_ticks are the raw servo-side values."
            ),
        }
        with open(self.out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def log(self, *, step_idx, state_rad, state_ticks,
            action_raw_rad, action_applied_rad, action_applied_ticks,
            gripper1_raw, gripper2_raw, chunk_remaining,
            cam0_frame=None, cam1_frame=None):
        if self.t_start is None:
            self.t_start = time.time()
        ts = time.time() - self.t_start

        self.rows.append({
            "timestamp":               float(ts),
            "step_idx":                int(step_idx),
            "chunk_remaining":         int(chunk_remaining),
            "observation.state":       np.asarray(state_rad,             dtype=np.float32).tolist(),
            "observation.state_ticks": np.asarray(state_ticks,           dtype=np.float32).tolist(),
            "action.raw":              np.asarray(action_raw_rad,        dtype=np.float32).tolist(),
            "action.applied":          np.asarray(action_applied_rad,    dtype=np.float32).tolist(),
            "action.applied_ticks":    np.asarray(action_applied_ticks,  dtype=np.float32).tolist(),
            "gripper1_raw":            float(gripper1_raw),
            "gripper2_raw":            float(gripper2_raw),
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
            fallback = self.out_dir / f"episode_{self.episode_idx:03d}.npz"
            print(f"Recorder: parquet write failed ({e}); saving npz instead.")
            np.savez(
                fallback,
                timestamp=np.array([r["timestamp"] for r in self.rows], dtype=np.float32),
                step_idx=np.array([r["step_idx"] for r in self.rows], dtype=np.int64),
                state=np.stack([np.array(r["observation.state"],          dtype=np.float32) for r in self.rows]),
                state_ticks=np.stack([np.array(r["observation.state_ticks"], dtype=np.float32) for r in self.rows]),
                action_raw=np.stack([np.array(r["action.raw"],            dtype=np.float32) for r in self.rows]),
                action_applied=np.stack([np.array(r["action.applied"],    dtype=np.float32) for r in self.rows]),
                action_applied_ticks=np.stack([np.array(r["action.applied_ticks"], dtype=np.float32) for r in self.rows]),
                gripper1_raw=np.array([r["gripper1_raw"] for r in self.rows], dtype=np.float32),
                gripper2_raw=np.array([r["gripper2_raw"] for r in self.rows], dtype=np.float32),
            )
            print(f"Recorder: wrote {len(self.rows)} rows -> {fallback}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    policy, preprocessor, postprocessor = load_act_policy(args.model, device)

    # Open both arm ports — talk to servos in raw ticks (no degree conversion).
    ph    = PacketHandler(0)
    print(f"Opening arm ports:  arm1={args.port1}  arm2={args.port2}")
    port1 = open_port(args.port1)
    port2 = open_port(args.port2)
    set_torque(ph, port1, True)
    set_torque(ph, port2, True)
    print("Torque enabled on both arms.")

    print(f"Opening cameras (cam0={args.cam0}, cam1={args.cam1})...")
    cap0 = open_camera(args.cam0)
    cap1 = open_camera(args.cam1)
    print(f"Cameras ready. step_scale={args.step_scale}  max_step={args.max_step} ticks")

    # Joint order: arm1_j1..j6 then arm2_j1..j6 — matches the training data.
    joint_keys = (
        [f"arm1_j{i}" for i in range(1, 7)] +
        [f"arm2_j{i}" for i in range(1, 7)]
    )

    # Initial state read — fall back to mid-tick if the bus stutters.
    ticks1 = read_ticks(ph, port1, np.full(6, 2048.0, dtype=np.float32))
    ticks2 = read_ticks(ph, port2, np.full(6, 2048.0, dtype=np.float32))

    recorder = None
    if args.record_dir is not None:
        recorder = TrajectoryRecorder(
            out_dir=args.record_dir,
            episode_idx=args.episode_idx,
            fps=args.fps,
            joint_keys=joint_keys,
            record_video=not args.no_record_video,
        )
        print(f"Recording to {args.record_dir} (episode {args.episode_idx:03d})")

    action_chunk = deque()
    chunk_size   = 10
    step_idx     = 0

    print(f"\nRunning at {args.fps} fps.  Ctrl-C to stop.\n")

    try:
        while True:
            t0 = time.time()

            # ── read joint state (ticks) ──
            ticks1 = read_ticks(ph, port1, ticks1)
            ticks2 = read_ticks(ph, port2, ticks2)
            state_ticks = np.concatenate([ticks1, ticks2]).astype(np.float32)
            state_rad   = state_ticks * TICKS_TO_RAD          # → radians for the model

            # ── cameras ──
            cam0_t = read_camera_tensor(cap0)
            cam1_t = read_camera_tensor(cap1)
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

            # ── inference (radians in, radians out) ──
            if not action_chunk:
                obs = {
                    "observation.images.cam0": cam0_t,
                    "observation.images.cam1": cam1_t,
                    "observation.state":       torch.from_numpy(state_rad),
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

            action_rad = action_chunk.popleft().astype(np.float32)   # 12-D, radians
            a1_rad = action_rad[:6].copy()
            a2_rad = action_rad[6:].copy()

            # ── gripper hysteresis (radian space) ──
            current1_rad = ticks1 * TICKS_TO_RAD
            current2_rad = ticks2 * TICKS_TO_RAD
            g1_raw = float(a1_rad[5])
            g2_raw = float(a2_rad[5])
            # a1_rad[5] = apply_gripper_hysteresis(g1_raw, float(current1_rad[5]))
            # a2_rad[5] = apply_gripper_hysteresis(g2_raw, float(current2_rad[5]))

            # ── radians → ticks, then delta-step in tick space ──
            target1_ticks = a1_rad * RAD_TO_TICKS
            target2_ticks = a2_rad * RAD_TO_TICKS

            delta1 = target1_ticks - ticks1
            delta2 = target2_ticks - ticks2
            step1  = np.clip(delta1 * args.step_scale, -args.max_step, args.max_step)
            step2  = np.clip(delta2 * args.step_scale, -args.max_step, args.max_step)
            applied1_ticks = ticks1 + step1
            applied2_ticks = ticks2 + step2

            # ── write to motors ──
            write_ticks(ph, port1, applied1_ticks)
            write_ticks(ph, port2, applied2_ticks)

            # bookkeeping for the recorder (everything in radians for comparability)
            applied_ticks = np.concatenate([applied1_ticks, applied2_ticks]).astype(np.float32)
            applied_rad   = applied_ticks * TICKS_TO_RAD

            if recorder is not None:
                recorder.log(
                    step_idx=step_idx,
                    state_rad=state_rad,
                    state_ticks=state_ticks,
                    action_raw_rad=action_rad,
                    action_applied_rad=applied_rad,
                    action_applied_ticks=applied_ticks,
                    gripper1_raw=g1_raw,
                    gripper2_raw=g2_raw,
                    chunk_remaining=len(action_chunk),
                    cam0_frame=cam0_rgb,
                    cam1_frame=cam1_rgb,
                )

            if step_idx % args.print_every == 0:
                print(f"[step {step_idx:05d}]  chunk_remaining={len(action_chunk)}")
                print(f"  arm1 cur ticks: {ticks1.astype(int)}")
                print(f"  arm1 raw rad  : {np.round(action_rad[:6], 3)}")
                print(f"  arm1 applied  : {applied1_ticks.astype(int)}")
                print(f"  arm2 cur ticks: {ticks2.astype(int)}")
                print(f"  arm2 raw rad  : {np.round(action_rad[6:], 3)}")
                print(f"  arm2 applied  : {applied2_ticks.astype(int)}")
                print(f"  gripper1 raw rad: {g1_raw:.3f}  cmd ticks: {int(applied1_ticks[5])}")
                print(f"  gripper2 raw rad: {g2_raw:.3f}  cmd ticks: {int(applied2_ticks[5])}")

            step_idx += 1
            dt = time.time() - t0
            time.sleep(max(1.0 / args.fps - dt, 0))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("Releasing torque and closing ports...")
        if recorder is not None:
            recorder.close()
        cap0.release()
        cap1.release()
        if args.show_cameras:
            cv2.destroyAllWindows()
        try:
            set_torque(ph, port1, False)
            set_torque(ph, port2, False)
            port1.closePort()
            port2.closePort()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()
