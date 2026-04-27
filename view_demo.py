"""
Parquet Demo Viewer
===================
Inspect and preview camera frames stored inside a demo Parquet file.

Usage:
    python view_demo.py --file data/demo.parquet          # summary + first frame
    python view_demo.py --file data/demo.parquet --play   # play back all frames
    python view_demo.py --file data/demo.parquet --episode 1 --play
    python view_demo.py --file data/demo.parquet --play --speed 0.5
"""

import argparse, sys, time
import numpy as np
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not found.  pip install pandas")

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found.  pip install opencv-python")


def decode_frame(jpeg_bytes) -> np.ndarray | None:
    if jpeg_bytes is None:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser(description="Inspect demo Parquet file")
    parser.add_argument("--file",    default=None,
                        help="Path to .parquet file (default: latest in dataset/)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode to view (default: all)")
    parser.add_argument("--play",    action="store_true",
                        help="Play back camera frames in a window")
    parser.add_argument("--speed",   type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0, try 0.5 for slow-mo)")
    parser.add_argument("--tile-width", type=int, default=320,
                        help="Display width per camera tile in pixels (default 320)")
    args = parser.parse_args()

    if args.file is None:
        files = sorted(Path("dataset/data").rglob("episode_*.parquet"))
        if not files:
            sys.exit("No parquet files found in dataset/data/")
        args.file = str(files[-1])
        print(f"  Auto-selected: {args.file}")

    # ── load ──
    print(f"\n  Loading {args.file} …")
    df = pd.read_parquet(args.file)

    if args.episode is not None:
        df = df[df["episode"] == args.episode]
        if df.empty:
            sys.exit(f"  No data found for episode {args.episode}")

    # ── summary ──
    cam_cols = [c for c in df.columns if c.endswith("_image")]
    episodes = sorted(df["episode"].unique())

    print(f"""
  ── Parquet summary ──────────────────────
  Rows       : {len(df)}
  Episodes   : {episodes}
  Columns    : {list(df.columns)}
  Camera cols: {cam_cols}
  Duration   : {df['t'].max():.2f}s  (last timestamp)
  ─────────────────────────────────────────""")

    for col in cam_cols:
        n_null = df[col].isna().sum()
        n_ok   = len(df) - n_null
        sizes  = df[col].dropna().apply(len)
        print(f"  {col}: {n_ok} frames saved, "
              f"{n_null} missing, "
              f"avg {sizes.mean()/1024:.1f} KB per frame")

    if not cam_cols:
        print("\n  No camera columns found in file.")
        return

    # ── quick decode check on first row ──
    first = df.iloc[0]
    for col in cam_cols:
        frame = decode_frame(first[col])
        if frame is not None:
            h, w = frame.shape[:2]
            print(f"\n  {col} first frame decoded OK — {w}×{h}")
        else:
            print(f"\n  {col} first frame could NOT be decoded")

    if not args.play:
        print("\n  Run with --play to preview frames in a window.\n")
        return

    # ── pre-decode all frames ──
    # Decode JPEG → ndarray up front so the display loop does zero decompression
    # work and can spend all its time on precise timing.
    target_w = args.tile_width
    target_h = target_w * 3 // 4   # assume 4:3 source

    print(f"\n  Pre-decoding {len(df)} frames … ", end="", flush=True)
    t_decode_start = time.time()

    records = []   # list of (t, episode, display_image)
    blank   = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    for _, row in df.iterrows():
        tiles = []
        for col in cam_cols:
            frame = decode_frame(row[col])
            if frame is None:
                tile = blank.copy()
            else:
                tile = cv2.resize(frame, (target_w, target_h),
                                  interpolation=cv2.INTER_LINEAR)
            # burn label directly into the tile
            cv2.putText(tile,
                        f"{col}  t={row['t']:.2f}s  ep{int(row['episode'])}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1, cv2.LINE_AA)
            tiles.append(tile)

        display = np.hstack(tiles)
        records.append((float(row["t"]), int(row["episode"]), display))

    print(f"done ({time.time() - t_decode_start:.1f}s)")
    print(f"  Playing back …  speed={args.speed}x  press q or ESC to quit\n")

    # ── playback loop ──
    # Instead of sleeping per-frame, we anchor to a wall-clock start time and
    # compute exactly when each frame should appear. This absorbs any drift from
    # imshow / waitKey overhead automatically.
    win = f"Demo: {args.file}  —  q/ESC quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    ep_wall_start = time.perf_counter()
    prev_ep       = records[0][1]

    for t_rec, ep, display in records:

        # Reset wall anchor at episode boundaries so episodes play independently.
        if ep != prev_ep:
            ep_wall_start = time.perf_counter()
            prev_ep = ep

        # When should this frame appear on the wall clock?
        target_wall = ep_wall_start + t_rec / args.speed

        # Sleep until just before the target, then busy-wait for precision.
        now = time.perf_counter()
        sleep_for = target_wall - now - 0.002   # wake 2 ms early
        if sleep_for > 0:
            time.sleep(sleep_for)
        while time.perf_counter() < target_wall:
            pass

        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()
    print("  Done.")


if __name__ == "__main__":
    main()