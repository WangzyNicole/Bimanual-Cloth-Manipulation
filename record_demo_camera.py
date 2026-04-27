"""
Manual Demo Recorder for SO-101 — Parquet Output
=================================================
Torque off on all joints. Move the arm by hand.
Joint positions + camera frames are saved into a single Parquet file.

Schema (one row per timestep):
  episode   : int   — episode index
  t         : float — time in seconds since episode start
  j1..j6    : int   — joint ticks for arm 1  (or arm1_j1..arm1_j6 / arm2_j1..arm2_j6)
  cam0_image: bytes — JPEG-encoded frame from camera 0
  cam1_image: bytes — JPEG-encoded frame from camera 1  (if present)
  ...

Usage:
  python record_demo_camera.py --port /dev/ttyACM0 --out data/demo.parquet
  python record_demo_camera.py --port /dev/ttyACM0 --port1 /dev/ttyACM1 \
                        --cams 0 1 --out data/demo.parquet
"""

import argparse, sys, time, tty, termios, threading, json
from pathlib import Path
from datetime import datetime

from scservo_sdk import PortHandler, PacketHandler

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found.  pip install opencv-python")

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow / pandas not found.  pip install pyarrow pandas")


# ── servo constants ───────────────────────────────────────────────────────────
PRESENT_ADDR = 56
TORQUE_ADDR  = 40

# ── servo helpers ─────────────────────────────────────────────────────────────

def open_port(name):
    p = PortHandler(name)
    if not p.openPort():
        raise RuntimeError(f"Cannot open {name}")
    p.setBaudRate(1_000_000)
    return p

def get_all_ticks(ph, port):
    ticks = []
    for i in range(1, 7):
        v, comm, _ = ph.read2ByteTxRx(port, i, PRESENT_ADDR)
        ticks.append(v if comm == 0 else -1)
    return ticks

def set_torque(ph, port, on):
    for i in range(1, 7):
        ph.write1ByteTxRx(port, i, TORQUE_ADDR, 1 if on else 0)

# ── keyboard helper ───────────────────────────────────────────────────────────

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

# ── camera manager ────────────────────────────────────────────────────────────

class CameraManager:
    """
    Captures frames from N cameras, each in its own background thread.

    One thread per camera means a stalled or slow USB camera cannot block
    the others — each camera runs its read() loop independently.
    """

    def __init__(self, cam_ids: list[int], hz: int, resolution=(640, 480)):
        self.cam_ids    = cam_ids
        self.interval   = 1.0 / hz
        self.resolution = resolution
        self._lock      = threading.Lock()
        self._latest    = {}          # cam_id → latest ndarray
        self._stale     = {}          # cam_id → consecutive missed frames
        self._stop      = threading.Event()
        self.caps       = {}
        self._threads   = []

        for cid in cam_ids:
            cap = cv2.VideoCapture(cid)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {cid}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.caps[cid]   = cap
            self._latest[cid] = None
            self._stale[cid]  = 0

        for cid in cam_ids:
            t = threading.Thread(target=self._loop, args=(cid,), daemon=True)
            self._threads.append(t)

    def start(self):
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        for cap in self.caps.values():
            cap.release()

    def grab_jpeg(self, cam_id: int, quality: int = 90) -> bytes | None:
        """Return the latest frame from cam_id encoded as JPEG bytes."""
        with self._lock:
            frame = self._latest.get(cam_id)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def stale_count(self, cam_id: int) -> int:
        """How many consecutive frames this camera has failed to deliver."""
        with self._lock:
            return self._stale.get(cam_id, 0)

    def _loop(self, cid: int):
        cap = self.caps[cid]
        while not self._stop.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            with self._lock:
                if ok:
                    self._latest[cid] = frame
                    self._stale[cid]  = 0
                else:
                    self._stale[cid] += 1
            elapsed = time.time() - t0
            time.sleep(max(0, self.interval - elapsed))

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Record SO-101 demos to a Parquet file.")
    parser.add_argument("--port",       default="/dev/ttyACM0",
                        help="Arm 1 serial port")
    parser.add_argument("--port1",      default=None,
                        help="Arm 2 serial port (bimanual)")
    parser.add_argument("--out",        default="dataset/data/chunk-000",
                        help="Output chunk directory (episode number auto-incremented)")
    parser.add_argument("--hz",         type=int, default=20,
                        help="Capture rate in Hz (default 20)")
    parser.add_argument("--cams",       type=int, nargs="+", default=[0],
                        help="Camera device IDs  e.g. --cams 0 1")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        metavar=("W", "H"))
    parser.add_argument("--jpeg-quality", type=int, default=90,
                        help="JPEG compression quality 1-100 (default 90)")
    args = parser.parse_args()

    chunk_dir  = Path(args.out)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    existing   = sorted(chunk_dir.glob("episode_*"))
    next_ep    = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0
    out_path   = chunk_dir / f"episode_{next_ep:06d}.parquet"
    resolution = tuple(args.resolution)

    # metadata folder sits parallel to the data folder
    meta_dir  = out_path.parent.parent / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / (out_path.stem + ".json")

    session_start = datetime.now().isoformat()

    # ── column names ──
    if args.port1:
        joint_cols = ([f"arm1_j{i}" for i in range(1, 7)] +
                      [f"arm2_j{i}" for i in range(1, 7)])
    else:
        joint_cols = [f"j{i}" for i in range(1, 7)]

    cam_cols = [f"cam{cid}_image" for cid in args.cams]
    all_cols  = ["episode", "t"] + joint_cols + cam_cols

    # ── pyarrow schema ──
    fields = [
        pa.field("episode", pa.int32()),
        pa.field("t",       pa.float32()),
    ]
    for col in joint_cols:
        fields.append(pa.field(col, pa.int32()))
    for col in cam_cols:
        fields.append(pa.field(col, pa.large_binary()))   # JPEG bytes

    schema = pa.schema(fields)

    # ── open hardware ──
    ph    = PacketHandler(0)
    port0 = open_port(args.port)
    port1 = open_port(args.port1) if args.port1 else None

    set_torque(ph, port0, False)
    if port1:
        set_torque(ph, port1, False)

    print(f"  Opening cameras {args.cams} …")
    cam_mgr = CameraManager(args.cams, hz=args.hz, resolution=resolution)
    cam_mgr.start()

    # Warm up cameras — give them a moment to deliver their first frame
    # before recording starts so we don't log None for the first few rows.
    print("  Warming up cameras ", end="", flush=True)
    for _ in range(20):
        time.sleep(0.05)
        print(".", end="", flush=True)
    print(" ready")

    print(f"""
═══════════════════════════════════════════
  SO-101 Demo Recorder  →  Parquet
  Output : {out_path}
  Rate   : {args.hz} fps
  Cameras: {args.cams}   {resolution[0]}×{resolution[1]}  q={args.jpeg_quality}
═══════════════════════════════════════════
  ENTER  : start / stop recording
  q      : quit and save
═══════════════════════════════════════════
""")

    # ── state ──
    recording     = False
    stop_flag     = threading.Event()
    episode       = 0
    rows          = []          # buffer for current episode
    episode_stats = []          # metadata per episode

    # open Parquet writer (append-friendly via ParquetWriter)
    pq_writer = pq.ParquetWriter(out_path, schema, compression="snappy")

    def flush_episode(episode_rows: list):
        """Convert buffered rows to an Arrow table and write one row group."""
        if not episode_rows:
            return
        # transpose list-of-dicts → dict-of-lists
        col_data = {col: [] for col in all_cols}
        for r in episode_rows:
            for col in all_cols:
                col_data[col].append(r[col])

        arrays = []
        for field in schema:
            arr = col_data[field.name]
            if pa.types.is_large_binary(field.type):
                arrays.append(pa.array(arr, type=pa.large_binary()))
            elif pa.types.is_float32(field.type):
                arrays.append(pa.array(arr, type=pa.float32()))
            else:
                arrays.append(pa.array(arr, type=pa.int32()))

        table = pa.table({f.name: a for f, a in zip(schema, arrays)},
                         schema=schema)
        pq_writer.write_table(table)

        duration = round(float(col_data["t"][-1]), 4) if col_data["t"] else 0
        n_frames = len(episode_rows)
        actual_hz = round(n_frames / duration, 1) if duration > 0 else 0

        # Per-camera stale frame count
        cam_missing = {}
        for col in cam_cols:
            cam_missing[col] = sum(1 for v in col_data[col] if v is None)

        episode_stats.append({
            "episode":    episode,
            "n_frames":   n_frames,
            "duration_s": duration,
            "actual_hz":  actual_hz,
            "cam_missing": cam_missing,
            "recorded_at": datetime.now().isoformat(),
        })

        missing_str = "  ".join(f"{c}: {v} missing" for c, v in cam_missing.items())
        print(f"\n  Episode {episode:03d} written — {n_frames} rows  "
              f"({duration:.2f}s  actual {actual_hz} hz)  {missing_str}")

    # ── background record loop ──
    def record_loop():
        t_start  = None
        interval = 1.0 / args.hz

        while not stop_flag.is_set():
            t0 = time.time()

            if recording:
                if t_start is None:
                    t_start = t0

                t = round(t0 - t_start, 4)

                joints = get_all_ticks(ph, port0)
                if port1:
                    joints += get_all_ticks(ph, port1)

                row = {"episode": episode, "t": t}
                for col, val in zip(joint_cols, joints):
                    row[col] = val
                for cid, col in zip(args.cams, cam_cols):
                    row[col] = cam_mgr.grab_jpeg(cid, quality=args.jpeg_quality)

                rows.append(row)
                n = len(rows)

                # Show stale warnings inline
                stale_parts = []
                for cid in args.cams:
                    s = cam_mgr.stale_count(cid)
                    if s > 2:
                        stale_parts.append(f"cam{cid}:stale({s})")
                stale_str = "  ⚠ " + " ".join(stale_parts) if stale_parts else ""

                print(f"\r  ● recording  {t:.2f}s  {n} frames{stale_str}   ",
                      end="", flush=True)
            else:
                t_start = None

            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    thread = threading.Thread(target=record_loop, daemon=True)
    thread.start()

    # ── key loop ──
    try:
        while True:
            k = get_key()

            if k in ('\r', '\n'):
                recording = not recording

                if recording:
                    episode += 1
                    rows.clear()
                    print(f"\n  ● RECORDING ep{episode:03d} — move the arm …")
                else:
                    recording = False
                    time.sleep(0.05)
                    flush_episode(rows)
                    rows.clear()

            elif k == 'q':
                if recording:
                    recording = False
                    time.sleep(0.05)
                    flush_episode(rows)
                break

    finally:
        stop_flag.set()
        thread.join()
        pq_writer.close()
        cam_mgr.stop()

        if port1:
            set_torque(ph, port1, False)
            port1.closePort()
        set_torque(ph, port0, False)
        port0.closePort()

        size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
        print(f"\n  Saved → {out_path}  ({size_mb:.1f} MB)  {episode} episode(s)")

        # ── write metadata ──
        metadata = {
            "session_start":  session_start,
            "session_end":    datetime.now().isoformat(),
            "parquet_file":   str(out_path),
            "hz":             args.hz,
            "resolution":     list(resolution),
            "jpeg_quality":   args.jpeg_quality,
            "ports":          {"arm1": args.port, "arm2": args.port1},
            "cameras":        args.cams,
            "joint_columns":  joint_cols,
            "camera_columns": cam_cols,
            "n_episodes":     episode,
            "total_frames":   sum(e["n_frames"] for e in episode_stats),
            "size_mb":        round(size_mb, 2),
            "episodes":       episode_stats,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata → {meta_path}\n")


if __name__ == "__main__":
    main()