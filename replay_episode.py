#!/usr/bin/env python3
"""
Replay a training episode from dataset/data/chunk-001 on the SO-101 bimanual arms.

The parquet files store raw servo ticks recorded at 20 Hz.  This script
reads those ticks and writes them directly as goal positions, reproducing
the original motion at real-time (or any scaled speed).

Usage:
    python replay_episode.py --episode 0
    python replay_episode.py --episode 5 --speed 0.5
    python replay_episode.py --episode 0 --chunk dataset/data/chunk-001 \\
        --port /dev/cu.usbmodem... --port1 /dev/cu.usbmodem...
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not found.  pip install pandas")

from scservo_sdk import PortHandler, PacketHandler

# Servo register addresses (SCS / FEETECH protocol)
TORQUE_ADDR        = 40
GOAL_POSITION_ADDR = 42
PRESENT_ADDR       = 56

# Default ports — update these to match your hardware
DEFAULT_PORT_ARM1 = "/dev/cu.usbmodem5AE60581371"   # RHS arm1
DEFAULT_PORT_ARM2 = "/dev/cu.usbmodem5AE60846081"   # LHS arm2


def open_port(name: str) -> PortHandler:
    p = PortHandler(name)
    if not p.openPort():
        raise RuntimeError(f"Cannot open port: {name}")
    p.setBaudRate(1_000_000)
    return p


def set_torque(ph: PacketHandler, port: PortHandler, on: bool):
    for servo_id in range(1, 7):
        ph.write1ByteTxRx(port, servo_id, TORQUE_ADDR, 1 if on else 0)


def read_positions(ph: PacketHandler, port: PortHandler) -> list[int]:
    positions = []
    for servo_id in range(1, 7):
        v, comm, _ = ph.read2ByteTxRx(port, servo_id, PRESENT_ADDR)
        positions.append(v if comm == 0 else 2048)  # 2048 = midpoint fallback
    return positions


def send_positions(ph: PacketHandler, port: PortHandler, ticks: list[int]):
    for servo_id, tick in enumerate(ticks, start=1):
        ph.write2ByteTxRx(port, servo_id, GOAL_POSITION_ADDR, int(tick))


def ramp_to_start(ph, port_arm1, port_arm2, target1, target2,
                  n_steps: int = 60, step_delay: float = 0.04):
    """Smoothly interpolate from current pose to the episode's first frame."""
    print("  Ramping to start position …")
    current1 = read_positions(ph, port_arm1)
    current2 = read_positions(ph, port_arm2)

    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp1 = [int(c + alpha * (t - c)) for c, t in zip(current1, target1)]
        interp2 = [int(c + alpha * (t - c)) for c, t in zip(current2, target2)]
        send_positions(ph, port_arm1, interp1)
        send_positions(ph, port_arm2, interp2)
        time.sleep(step_delay)

    print("  At start position.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay a training episode on the SO-101 bimanual arms.")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode number to replay (default: 0)")
    parser.add_argument("--chunk",   default="dataset/data/chunk-001",
                        help="Chunk directory containing the parquet files")
    parser.add_argument("--port",    default=DEFAULT_PORT_ARM1,
                        help="Serial port for arm1 / RHS")
    parser.add_argument("--port1",   default=DEFAULT_PORT_ARM2,
                        help="Serial port for arm2 / LHS")
    parser.add_argument("--speed",   type=float, default=1.0,
                        help="Playback speed multiplier (0.5 = half speed, 2.0 = double)")
    parser.add_argument("--no-ramp", action="store_true",
                        help="Skip the smooth ramp to the start position")
    args = parser.parse_args()

    # ── load parquet ──────────────────────────────────────────────────────────
    chunk_dir = Path(args.chunk)
    ep_file   = chunk_dir / f"episode_{args.episode:06d}.parquet"
    if not ep_file.exists():
        available = sorted(chunk_dir.glob("episode_*.parquet"))
        n = len(available)
        sys.exit(
            f"Episode file not found: {ep_file}\n"
            f"  {n} episodes available (0 – {n - 1}) in {chunk_dir}"
        )

    print(f"\n  Loading {ep_file} …")
    df = pd.read_parquet(ep_file)

    # parquet may contain multiple episode rows — keep only the first episode id
    if "episode" in df.columns:
        df = df[df["episode"] == df["episode"].iloc[0]]

    arm1_cols = [f"arm1_j{i}" for i in range(1, 7)]
    arm2_cols = [f"arm2_j{i}" for i in range(1, 7)]
    for col in arm1_cols + arm2_cols:
        if col not in df.columns:
            sys.exit(
                f"Missing column '{col}'.  Is this a bimanual (arm1_j*/arm2_j*) parquet file?"
            )

    timestamps = df["t"].to_numpy(dtype=float)
    arm1_ticks = df[arm1_cols].to_numpy(dtype=int)
    arm2_ticks = df[arm2_cols].to_numpy(dtype=int)

    n_frames = len(df)
    duration = float(timestamps[-1])
    print(f"  Episode {args.episode}: {n_frames} frames, {duration:.2f}s")
    print(f"  Playback speed: {args.speed}x  → {duration / args.speed:.2f}s wall time\n")

    # ── connect hardware ──────────────────────────────────────────────────────
    ph       = PacketHandler(0)
    port_a1  = open_port(args.port)
    port_a2  = open_port(args.port1)

    set_torque(ph, port_a1, True)
    set_torque(ph, port_a2, True)
    print("  Torque enabled on both arms.")

    try:
        # ── ramp to episode start ─────────────────────────────────────────────
        if not args.no_ramp:
            ramp_to_start(ph, port_a1, port_a2,
                          list(arm1_ticks[0]), list(arm2_ticks[0]))
            # brief pause so the user can see the start pose
            time.sleep(0.5)

        # ── replay ───────────────────────────────────────────────────────────
        print("  Replaying …  Ctrl+C to stop\n")
        wall_start = time.perf_counter()

        for idx in range(n_frames):
            t_rec  = timestamps[idx]
            t_wall = wall_start + t_rec / args.speed

            # sleep most of the wait, then busy-wait the last 2 ms for precision
            slack = t_wall - time.perf_counter() - 0.002
            if slack > 0:
                time.sleep(slack)
            while time.perf_counter() < t_wall:
                pass

            send_positions(ph, port_a1, arm1_ticks[idx])
            send_positions(ph, port_a2, arm2_ticks[idx])

            if idx % 20 == 0 or idx == n_frames - 1:
                pct = 100.0 * (idx + 1) / n_frames
                print(f"\r  [{pct:5.1f}%]  frame {idx + 1}/{n_frames}  t={t_rec:.2f}s   ",
                      end="", flush=True)

        print(f"\r  [100.0%]  Done — {n_frames} frames replayed.                  ")

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    finally:
        set_torque(ph, port_a1, False)
        set_torque(ph, port_a2, False)
        port_a1.closePort()
        port_a2.closePort()
        print("  Torque disabled. Ports closed.\n")


if __name__ == "__main__":
    main()
