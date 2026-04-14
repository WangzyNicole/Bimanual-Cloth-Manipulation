"""
Camera Preview
==============
Opens one or more cameras and shows their live feeds in a tiled window.
Use this to verify camera IDs, angles, and resolution before recording.

Usage:
    python camera_preview.py                    # try camera 0
    python camera_preview.py --cams 0 1         # two cameras side by side
    python camera_preview.py --cams 0 1 2       # three cameras
    python camera_preview.py --resolution 1280 720

Controls:
    q / ESC   quit
    s         save a snapshot of the current tiled view
    f         toggle fullscreen
"""

import argparse, sys, time
from pathlib import Path
from datetime import datetime

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("OpenCV not found.  pip install opencv-python")


def open_cameras(cam_ids: list[int], resolution: tuple) -> list:
    caps = []
    for cid in cam_ids:
        cap = cv2.VideoCapture(cid)
        if not cap.isOpened():
            print(f"  [warn] Cannot open camera {cid} — skipping")
            caps.append(None)
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        caps.append(cap)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  cam{cid}  opened  →  {actual_w}×{actual_h}")
    return caps


def tile_frames(frames: list, labels: list, target_w: int = 640) -> np.ndarray:
    """Stack frames horizontally, each rescaled to target_w wide."""
    tiles = []
    for frame, label in zip(frames, labels):
        if frame is None:
            tile = np.zeros((target_w * 3 // 4, target_w, 3), dtype=np.uint8)
            cv2.putText(tile, "no signal", (target_w // 2 - 60, target_w * 3 // 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        else:
            h, w = frame.shape[:2]
            new_h = int(h * target_w / w)
            tile  = cv2.resize(frame, (target_w, new_h))

        # label bar at top
        bar = np.zeros((28, tile.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        tiles.append(np.vstack([bar, tile]))

    return np.hstack(tiles)


def main():
    parser = argparse.ArgumentParser(description="Live camera preview")
    parser.add_argument("--cams",       type=int, nargs="+", default=[0],
                        help="Camera device IDs (default: 0)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        metavar=("W", "H"))
    parser.add_argument("--tile-width", type=int, default=640,
                        help="Display width per camera tile in pixels (default 640)")
    args = parser.parse_args()

    resolution = tuple(args.resolution)
    win_name   = "Camera Preview  —  q/ESC quit · s snapshot · f fullscreen"
    fullscreen  = False

    print(f"\n  Opening cameras {args.cams} at {resolution[0]}×{resolution[1]} …\n")
    caps   = open_cameras(args.cams, resolution)
    labels = [f"cam{cid}" for cid in args.cams]

    if all(c is None for c in caps):
        sys.exit("No cameras could be opened.")

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print("\n  Preview running.  Press q or ESC to quit.\n")

    fps_time = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        frames = []
        for cap in caps:
            if cap is None:
                frames.append(None)
            else:
                ok, frame = cap.read()
                frames.append(frame if ok else None)

        # FPS counter
        fps_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_display = fps_count / (now - fps_time)
            fps_count   = 0
            fps_time    = now

        # build tiled view
        tiled = tile_frames(frames, labels, target_w=args.tile_width)

        # overlay FPS
        cv2.putText(tiled, f"{fps_display:.1f} fps",
                    (tiled.shape[1] - 90, tiled.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        cv2.imshow(win_name, tiled)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):          # q or ESC
            break

        elif key == ord('s'):              # snapshot
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"snapshot_{ts}.jpg"
            cv2.imwrite(path, tiled)
            print(f"  Snapshot saved → {path}")

        elif key == ord('f'):              # toggle fullscreen
            fullscreen = not fullscreen
            flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    for cap in caps:
        if cap:
            cap.release()
    cv2.destroyAllWindows()
    print("  Preview closed.")


if __name__ == "__main__":
    main()
