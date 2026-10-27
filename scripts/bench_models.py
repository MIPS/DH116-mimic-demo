#!/usr/bin/env python3
"""Estimate hand-pipeline inference time on whatever machine this runs on.

Needs only `onnxruntime` and `numpy` (pip install onnxruntime numpy); no OpenCV, no build.
Times the palm detector and the landmark network with ONNX Runtime on random input
(convolution cost does not depend on pixel values), for one or several thread counts,
and turns that into a per-frame estimate of the model part of the pipeline.

Camera capture, colour conversion, crop and drawing are NOT included; they are small next to
the networks (about 0.5 ms on a desktop CPU).

Examples:
  python3 scripts/bench_models.py
  python3 scripts/bench_models.py --threads 1,2,4,8 --iters 200
  taskset -c 0-7 python3 scripts/bench_models.py --threads 8     # pin to specific cores
"""
import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as exc:
    sys.exit(f"{exc}\nInstall with: pip install onnxruntime numpy")

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"palm": "hand_palm.onnx", "landmark": "hand_landmark_detector.onnx"}
ORT_TYPES = {"tensor(float)": np.float32, "tensor(uint8)": np.uint8, "tensor(int8)": np.int8,
             "tensor(float16)": np.float16}


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.split(":")[0].strip() in ("model name", "uarch", "isa"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def make_input(session):
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    dtype = ORT_TYPES.get(inp.type, np.float32)
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.floating):
        data = rng.random(shape, dtype=np.float32).astype(dtype)
    else:
        data = rng.integers(0, 255, shape).astype(dtype)
    return inp.name, data


def bench(path, threads, warmup, iters, no_spin):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if no_spin:
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    name, data = make_input(session)
    for _ in range(warmup):
        session.run(None, {name: data})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, {name: data})
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {
        "mean": statistics.fmean(times),
        "median": times[len(times) // 2],
        "p95": times[min(len(times) - 1, int(len(times) * 0.95))],
        "min": times[0],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models",
                    help="directory with hand_palm.onnx and hand_landmark_detector.onnx")
    ap.add_argument("--threads", default=None,
                    help="comma-separated intra-op thread counts (default: 1 and all visible cores)")
    ap.add_argument("--iters", type=int, default=100, help="timed runs per model and thread count")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--det-every", type=int, default=15, help="palm re-detection interval used for the estimate")
    ap.add_argument("--no-spin", action="store_true",
                    help="disable ORT thread spinning (default is on, like hand_cam; off saves CPU but adds wake-up latency)")
    args = ap.parse_args()

    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    counts = sorted({int(t) for t in args.threads.split(",")}) if args.threads else sorted({1, cores})

    paths = {k: args.model_dir / v for k, v in MODELS.items()}
    for p in paths.values():
        if not p.is_file():
            sys.exit(f"model not found: {p}")

    print(f"CPU        : {cpu_name()}")
    print(f"Machine    : {platform.machine()}, {cores} usable cores, Python {platform.python_version()}")
    print(f"ORT        : {ort.__version__}, providers {ort.get_available_providers()}")
    print(f"Runs       : {args.warmup} warmup + {args.iters} timed, spinning {'off' if args.no_spin else 'on'}")
    print()

    header = f"{'threads':>7} | {'model':<9} | {'median':>8} {'mean':>8} {'p95':>8} {'min':>8}  (ms)"
    print(header)
    print("-" * len(header))

    results = {}
    for n in counts:
        for key, path in paths.items():
            r = bench(path, n, args.warmup, args.iters, args.no_spin)
            results[(n, key)] = r
            print(f"{n:>7} | {key:<9} | {r['median']:8.2f} {r['mean']:8.2f} {r['p95']:8.2f} {r['min']:8.2f}")

    print()
    print(f"Estimated model time per frame (palm every {args.det_every} frames, landmark every frame, 1 hand):")
    print(f"{'threads':>7} | {'steady':>8} | {'average':>8} | {'spike':>8} | {'avg fps':>8}")
    for n in counts:
        palm = results[(n, "palm")]["median"]
        lm = results[(n, "landmark")]["median"]
        avg = lm + palm / args.det_every
        print(f"{n:>7} | {lm:7.2f}ms | {avg:7.2f}ms | {palm + lm:7.2f}ms | {1000 / avg:8.1f}")
    print("steady = landmark only; spike = frame that also runs the palm net.")
    print("Tip: if more threads is not faster, try a lower count or pin to the fast cores with taskset.")


if __name__ == "__main__":
    main()
