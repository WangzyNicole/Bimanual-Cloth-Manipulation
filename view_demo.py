"""
Parquet Demo Viewer
===================
Inspect and preview camera frames stored inside a demo Parquet file.

Usage:
    python view_demo.py --file data/demo.parquet          # summary + first frame
    python view_demo.py --file data/demo.parquet --play   # play back all frames
    python view_demo.py --file data/demo.parquet --episode 1 --play
"""

import argparse, sys, time
import numpy as np

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
    parser.add_argument("--file",    required=True, help="Path to .parquet file")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode to view (default: all)")
    parser.add_argument("--play",    action="store_true",
                        help="Play back camera frames in a window")
    args = parser.parse_args()

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

    # ── playback ──
    print("\n  Playing back …  press q or ESC to quit\n")
    fps      = 20
    interval = 1.0 / fps

    for _, row in df.iterrows():
        t0     = time.time()
        tiles  = []

        for col in cam_cols:
            frame = decode_frame(row[col])
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # label
            cv2.putText(frame, f"{col}  t={row['t']:.2f}s  ep{int(row['episode'])}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)
            tiles.append(frame)

        display = np.hstack(tiles)
        cv2.imshow("Demo playback  —  q / ESC to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

    cv2.destroyAllWindows()
    print("  Done.")


if __name__ == "__main__":
    main()
