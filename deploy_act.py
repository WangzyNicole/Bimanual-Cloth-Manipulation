"""
ACT Policy Deployment — Bimanual SO-100
========================================
Runs the pretrained ACT model on two SO-100 follower arms with two cameras.

This version uses incremental delta stepping (the version where arms
slowly moved down correctly). Gripper thresholds tuned to actual output range.

Usage:
    python deploy_act.py --model ./pretrained_model --cam0 0 --cam1 1

Controls:
    Ctrl-C   stop and release torque
"""

import argparse
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
    # Warm up until a real frame arrives
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
# Based on observed raw output range of 2.5–3.1 during deployment.
# Model rarely exceeds 3.5 or goes below 2.3 — thresholds set just outside
# observed range so gripper holds by default until model strongly signals.

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
    """
    Move incrementally toward the predicted target each tick.
    Gripper uses hysteresis instead of incremental stepping.
    """
    delta = action - current
    step  = np.clip(delta * step_scale, -max_step, max_step)
    target = {}
    for i, k in enumerate(joint_keys):
        if "gripper" in k:
            target[k] = apply_gripper_hysteresis(action[i], current[i])
        else:
            target[k] = float(current[i] + step[i])
    return target


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

            if args.show_cameras:
                disp0 = cv2.cvtColor(
                    cam0_t.permute(1, 2, 0).numpy().astype(np.uint8),
                    cv2.COLOR_RGB2BGR)
                disp1 = cv2.cvtColor(
                    cam1_t.permute(1, 2, 0).numpy().astype(np.uint8),
                    cv2.COLOR_RGB2BGR)
                cv2.imshow("cam0 | cam1", np.hstack([disp0, disp1]))
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

            if step_idx % 20 == 0:
                print(f"[step {step_idx:05d}]  chunk_remaining={len(action_chunk)}")
                print(f"  arm1  cur: {np.round(current1, 2)}")
                print(f"  arm1  tgt: {np.round(action1,  2)}")
                print(f"  arm2  cur: {np.round(current2, 2)}")
                print(f"  arm2  tgt: {np.round(action2,  2)}")
                print(f"  gripper1 raw: {action1[-1]:.3f}  cmd: {target1['gripper.pos']:.2f}")
                print(f"  gripper2 raw: {action2[-1]:.3f}  cmd: {target2['gripper.pos']:.2f}")

            step_idx += 1

            dt = time.time() - t0
            time.sleep(max(1.0 / args.fps - dt, 0))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("Releasing torque and disconnecting...")
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