import torch
import numpy as np
import time
import cv2
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from safetensors.torch import load_file

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


# ===== CONFIG =====
MODEL_PATH = str(Path("/Users/kronos.di.vlad/Downloads/2951k-hw2/pretrained_act_model").resolve())

PORT1 = "/dev/cu.usbmodem5AE60581371"  # RHS arm named arm1
PORT2 = "/dev/cu.usbmodem5AE60846081"  # LHS arm named arm2

CAM0_INDEX = 1  # RHS camera
CAM1_INDEX = 2  # LHS camera

FPS = 10


# ===== LOAD POLICY =====
cfg = PreTrainedConfig.from_pretrained(MODEL_PATH)
cfg.pretrained_path = MODEL_PATH
cfg.device = "mps" if torch.backends.mps.is_available() else "cpu"

PolicyCls = get_policy_class("act")
policy = PolicyCls(cfg)

state_dict = load_file(str(Path(MODEL_PATH) / "model.safetensors"))
policy.load_state_dict(state_dict)

policy.to(cfg.device)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=MODEL_PATH,
    dataset_stats=None,
    preprocessor_overrides={
        "device_processor": {"device": cfg.device},
    },
)

print("ACT policy loaded.")


# ===== CONNECT ROBOTS =====
robot1 = SO100Follower(SO100FollowerConfig(port=PORT1, id="arm1", use_degrees=True))
robot2 = SO100Follower(SO100FollowerConfig(port=PORT2, id="arm2", use_degrees=True))

print("Connecting robots...")
robot1.connect()
robot2.connect()
print("Connected.")


# ===== CAMERA SETUP =====
cap0 = cv2.VideoCapture(CAM0_INDEX, cv2.CAP_AVFOUNDATION)
cap1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_AVFOUNDATION)

cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap0.isOpened() or not cap1.isOpened():
    raise RuntimeError("Failed to open cameras")


# ===== JOINT ORDER (DO NOT SORT) =====
obs1 = robot1.get_observation()
obs2 = robot2.get_observation()

joint_keys1 = [k for k in obs1 if k.endswith(".pos")]
joint_keys2 = [k for k in obs2 if k.endswith(".pos")]

print("Arm1 joints:", joint_keys1)
print("Arm2 joints:", joint_keys2)


# # ===== MAIN LOOP =====
step = 0

while True:
    t0 = time.time()

    # --- robot state ---
    obs1 = robot1.get_observation()
    obs2 = robot2.get_observation()

    current1 = np.array([obs1[k] for k in joint_keys1], dtype=np.float32)
    current2 = np.array([obs2[k] for k in joint_keys2], dtype=np.float32)

    state = np.concatenate([current1, current2]).astype(np.float32)

    # --- camera frames ---
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 or not ret1:
        raise RuntimeError("Camera read failed")

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    cam0 = torch.from_numpy(frame0).permute(2, 0, 1).float()
    cam1 = torch.from_numpy(frame1).permute(2, 0, 1).float()

    # --- build policy input ---
    obs = {
        "observation.images.cam0": cam0,
        "observation.images.cam1": cam1,
        "observation.state": torch.from_numpy(state),
    }

    # --- ACT inference ---
    policy_input = preprocessor(obs)
    policy_output = policy.select_action(policy_input)
    action = postprocessor(policy_output)

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    action = np.asarray(action)
    if action.ndim == 3:
        action = action[0, 0]
    elif action.ndim == 2:
        action = action[0]

    # --- split policy output ---
    action1 = action[:6]
    action2 = action[6:]

    # --- incremental control toward ACT target ---
    delta1 = action1 - current1
    delta2 = action2 - current2

    STEP_SCALE = 0.5   # try 0.5 first; increase later if motion is too small
    MAX_STEP = 2.0     # max degrees per loop

    step1 = np.clip(delta1 * STEP_SCALE, -MAX_STEP, MAX_STEP)
    step2 = np.clip(delta2 * STEP_SCALE, -MAX_STEP, MAX_STEP)

    # target1 = {k: float(current1[i] + step1[i]) for i, k in enumerate(joint_keys1)}
    # target2 = {k: float(current2[i] + step2[i]) for i, k in enumerate(joint_keys2)}

    # --- hysteresis thresholds (tune if needed) ---
    GRIPPER_OPEN_THRESHOLD = 3.6
    GRIPPER_CLOSE_THRESHOLD = 3.3

    GRIPPER_OPEN_POS = 4.5    # fully open
    GRIPPER_CLOSE_POS = 2.0   # fully closed

    target1 = {}
    target2 = {}

    # --- arm1 (RHS) ---
    for i, k in enumerate(joint_keys1):
        if "gripper" in k:
            val = action1[i]

            if val > GRIPPER_OPEN_THRESHOLD:
                target1[k] = GRIPPER_OPEN_POS
            elif val < GRIPPER_CLOSE_THRESHOLD:
                target1[k] = GRIPPER_CLOSE_POS
            else:
                target1[k] = float(current1[i])  # hold
        else:
            target1[k] = float(current1[i] + step1[i])

    # --- arm2 (LHS) ---
    for i, k in enumerate(joint_keys2):
        if "gripper" in k:
            val = action2[i]

            if val > GRIPPER_OPEN_THRESHOLD:
                target2[k] = GRIPPER_OPEN_POS
            elif val < GRIPPER_CLOSE_THRESHOLD:
                target2[k] = GRIPPER_CLOSE_POS
            else:
                target2[k] = float(current2[i])  # hold
        else:
            target2[k] = float(current2[i] + step2[i])

    print(
    "arm1 gripper current/target:",
    obs1["gripper.pos"], target1["gripper.pos"],
    " | arm2 gripper current/target:",
    obs2["gripper.pos"], target2["gripper.pos"]
    )

    # --- send to robot ---
    robot1.send_action(target1)
    robot2.send_action(target2)

    # --- occasional debug ---
    if step % 20 == 0:
        print("arm1 current:", np.round(current1, 2))
        print("arm1 target :", np.round(action1, 2))
        print("arm1 delta  :", np.round(delta1, 2))
        print("arm2 current:", np.round(current2, 2))
        print("arm2 target :", np.round(action2, 2))
        print("arm2 delta  :", np.round(delta2, 2))
        print("----")

    step += 1

    # --- loop timing ---
    dt = time.time() - t0
    time.sleep(max(1.0 / FPS - dt, 0))