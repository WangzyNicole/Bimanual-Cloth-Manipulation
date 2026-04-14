# Bimanual Cloth Manipulation

Brown University CS — Tabitha Oanda, Nicole Wang, Xiaoyi Liu

## Hardware
- SO-101 bimanual robot (2 arms, 6 DOF each)
- 2 wrist-mounted cameras
- Silicone gripper pads for fabric grasp

## Setup
```bash
git clone https://github.com/WangzyNicole/Bimanual-Cloth-Manipulation.git
cd Bimanual-Cloth-Manipulation
pip install -r requirements.txt
```

## Data Collection
```bash
python record_demo_camera.py \
  --port /dev/ttyACM0 --port1 /dev/ttyACM1 \
  --cams 0 1 --out data/demo.parquet
```
Press ENTER to start/stop each episode. Press q to quit and save.

## Camera Live View
```bash
python camera_preview.py --cams 0 1
```

## Replay Recorded Episodes
```bash
python view_demo.py --file data/demo.parquet --play
```