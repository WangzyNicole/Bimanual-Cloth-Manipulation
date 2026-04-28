# Bimanual Cloth Manipulation

Brown University CS — Tabitha Oanda, Nicole Wang, Xiaoyi Liu

## Hardware

- SO-101 bimanual robot (2 × SO-100 follower arms, 6 DOF each)
- 2 wrist-mounted USB cameras (cam0 = RHS, cam1 = LHS)
- Silicone gripper pads for fabric grasp

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/WangzyNicole/Bimanual-Cloth-Manipulation.git
cd Bimanual-Cloth-Manipulation
pip install -r requirements.txt
```

> **Apple Silicon (MPS):** PyTorch MPS is supported and selected automatically.  
> **CUDA:** If running on a Linux/Windows GPU machine, install a CUDA-enabled torch build before running `pip install -r requirements.txt`:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Finding your serial ports

Each arm connects over USB serial.  Find your port names before running any script:

**macOS:**
```bash
ls /dev/cu.usbmodem*
```

**Linux:**
```bash
ls /dev/ttyACM*   # or /dev/ttyUSB*
```

Pass the discovered ports via `--port` (arm1 / RHS) and `--port1` (arm2 / LHS).  
The scripts default to the ports used during development:
```
arm1 (RHS): /dev/cu.usbmodem5AE60581371
arm2 (LHS): /dev/cu.usbmodem5AE60846081
```

---

## Finding your camera indices

```bash
python camera_preview.py --cams 0 1
```

Try different integer indices (0, 1, 2, …) until both wrist cameras appear.  
Pass the discovered indices via `--cam0` / `--cam1` (or `--cams`) in the scripts below.

---

## Calibration

Calibration files for our hardware are already committed as `arm1.json` and `arm2.json`.  
If you are setting up a **new pair of arms**, re-run calibration for each:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5AE60581371 \
  --robot.id=arm1

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5AE60846081 \
  --robot.id=arm2
```

This overwrites `arm1.json` / `arm2.json` in the project root.

---

## Data Collection

Torque is disabled so you can move the arms by hand.  Joint positions and camera frames are saved to a Parquet file.

```bash
python record_demo_camera.py \
  --port  /dev/cu.usbmodem5AE60581371 \
  --port1 /dev/cu.usbmodem5AE60846081 \
  --cams 0 1 \
  --out dataset/data/chunk-001
```

Controls:
- **ENTER** — start / stop recording an episode
- **q** — quit and save

Each episode is appended to the chunk directory as `episode_XXXXXX.parquet`.

---

## Replay a Recorded Episode

Play back camera footage only (no robot movement):

```bash
python view_demo.py --file dataset/data/chunk-001/episode_000000.parquet --play
python view_demo.py --file dataset/data/chunk-001/episode_000000.parquet --play --speed 0.5
```

Physically replay the arm motion from a training example:

```bash
python replay_episode.py --episode 0

# half speed, custom ports
python replay_episode.py --episode 5 --speed 0.5 \
  --port /dev/cu.usbmodem... --port1 /dev/cu.usbmodem...
```

> The arms ramp smoothly to the start pose before replaying.  Press **Ctrl-C** to stop; torque is released automatically.

---

## Deploy the ACT Policy

The pretrained model lives in `pretrained_model/`.  
The model was trained on joint positions in **radians** (`ticks × π/2048`).  
`deploy_act.py` handles the tick ↔ radian conversion automatically.

```bash
python deploy_act.py --model ./pretrained_model --cam0 0 --cam1 1
```

If your serial ports differ from the defaults, pass them explicitly:

```bash
python deploy_act.py --model ./pretrained_model --cam0 0 --cam1 1 \
  --port1 /dev/tty.usbmodem... \
  --port2 /dev/tty.usbmodem...
```

Record a deployment run to disk for analysis:

```bash
python deploy_act.py --model ./pretrained_model --cam0 0 --cam1 1 \
  --record-dir ./runs/$(date +%Y%m%d_%H%M%S)
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--fps` | 20 | Control loop frequency |
| `--step-scale` | 0.5 | Fraction of delta applied per tick |
| `--max-step` | 50 | Max servo ticks moved per joint per loop |
| `--show-cameras` | off | Display live camera feed |
| `--record-dir` | none | Directory to save parquet + MP4 |

Press **Ctrl-C** to stop; torque is released automatically.

---

## Dataset Format

Raw demos are stored as Apache Parquet files under `dataset/data/chunk-*/`.

| Column | Type | Description |
|---|---|---|
| `episode` | int32 | Episode index |
| `t` | float32 | Time in seconds since episode start |
| `arm1_j1` … `arm1_j6` | int32 | Arm 1 servo ticks (IDs 1–6) |
| `arm2_j1` … `arm2_j6` | int32 | Arm 2 servo ticks (IDs 1–6) |
| `cam0_image` | bytes | JPEG-encoded frame from cam0 |
| `cam1_image` | bytes | JPEG-encoded frame from cam1 |

Joint order within each arm: shoulder\_pan, shoulder\_lift, elbow\_flex, wrist\_flex, wrist\_roll, gripper.

---

## Project Structure

```
Bimanual-Cloth-Manipulation/
├── pretrained_model/          ACT policy weights + normalization stats
├── dataset/
│   ├── data/chunk-001/        170 training episodes (Parquet)
│   ├── videos/chunk-000/      Camera videos for first 25 episodes
│   └── meta/                  Dataset metadata (info.json, stats.json, episodes.jsonl)
├── arm1.json / arm2.json      Per-arm servo calibration
├── deploy_act.py              Run the ACT policy on the real robot
├── replay_episode.py          Replay a training episode on the arms
├── record_demo_camera.py      Collect new demonstrations
├── view_demo.py               Inspect / preview recorded Parquet episodes
├── camera_preview.py          Live camera feed
└── requirements.txt
```
