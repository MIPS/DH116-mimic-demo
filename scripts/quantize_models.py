#!/usr/bin/env python3
"""Quantize the two hand models to int8 (static QDQ) and compare them against fp32.

python scripts/quantize_models.py --convert models/lite/*.tflite --nchw

Dev-host tool, not needed on the target: pip install onnxruntime numpy onnx opencv-python-headless
It writes models/int8/*.onnx; copy that directory to the device and time it with
  python3 scripts/bench_models.py --model-dir models/int8

Calibration frames come from --source (images, video files, or a camera index) and, when only a
few frames are available, are augmented (rotation, scale, shift, brightness, flip). For a
trustworthy accuracy number use a real clip of the hand moving through its full range.

The report compares int8 with fp32 on the same frames:
  palm     : score and box-centre difference of the best detection
  landmark : world-landmark error in mm (on crops taken from the fp32 palm result) and presence
This is a proxy; the final check is the six joint features on a recorded clip.

Convert mode (--convert) turns TensorFlow Lite / TensorFlow models into ONNX and verifies them:
  python3 scripts/quantize_models.py --convert models/lite/*.tflite --out-dir models/converted
Accepts .tflite files, SavedModel directories and .keras/.h5 files. Each result is run next to the
original framework on the same inputs (random, plus frames from --source if given) and the outputs
are compared; the exit code is non-zero if any model fails. Needs a Python version TensorFlow
supports (3.9-3.12), e.g.: pip install tensorflow-cpu tf2onnx onnx onnxruntime numpy opencv-python-headless
Use --nchw to expose the image input as NCHW like models/hand_*.onnx (TFLite models are NHWC).
"""
import argparse
import math
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
except ImportError as exc:
    sys.exit(f"{exc}\nInstall with: pip install onnxruntime numpy onnx opencv-python-headless")

ROOT = Path(__file__).resolve().parent.parent
PALM, LM = 192, 224
ROI_SCALE, ROI_SHIFT = 2.9, 0.5  # keep in sync with src/hand_cam.cpp


# --------------------------------------------------------------------------- frames
def load_frames(sources, count, seed):
    rng = np.random.default_rng(seed)
    raw = []
    for src in sources:
        if src.isdigit():
            cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
            for _ in range(10):
                cap.read()
            for _ in range(count):
                ok, f = cap.read()
                if ok:
                    raw.append(f)
                time.sleep(0.1)
            cap.release()
        elif Path(src).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            f = cv2.imread(src)
            if f is not None:
                raw.append(f)
        else:
            cap = cv2.VideoCapture(src)
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            step = max(1, len(frames) // count)
            raw += frames[::step][:count]
    if not raw:
        sys.exit("no frames could be read from --source")
    frames = [cv2.resize(f, (640, 480)) for f in raw]
    if len(frames) >= count:
        return frames[:count]
    out = list(frames)
    while len(out) < count:  # augment so a handful of stills still give a spread of poses and lighting
        f = frames[int(rng.integers(len(frames)))]
        m = cv2.getRotationMatrix2D((320 + rng.uniform(-80, 80), 240 + rng.uniform(-60, 60)),
                                    rng.uniform(-40, 40), rng.uniform(0.6, 1.2))
        g = cv2.warpAffine(f, m, (640, 480), borderMode=cv2.BORDER_REFLECT)
        g = cv2.convertScaleAbs(g, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))
        out.append(cv2.flip(g, 1) if rng.random() < 0.5 else g)
    return out


# ----------------------------------------------------- fp32 pipeline (mirrors hand_cam.cpp)
def palm_input(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = PALM / max(h, w)
    nh, nw = int(h * s), int(w * s)
    py, px = (PALM - nh) // 2, (PALM - nw) // 2
    pad = np.zeros((PALM, PALM, 3), np.uint8)
    pad[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh))
    return pad.transpose(2, 0, 1)[None].astype(np.float32) / 255.0, (px, py, s)


def best_roi(det, pad):
    det = det[det[:, 0] >= 0.5]
    if not len(det):
        return None
    d = det[np.argmax(det[:, 0])]
    px, py, s = pad
    cx, cy = (d[1] * PALM - px) / s, (d[2] * PALM - py) / s
    k0 = ((d[4] * PALM - px) / s, (d[5] * PALM - py) / s)
    k2 = ((d[6] * PALM - px) / s, (d[7] * PALM - py) / s)
    sz = d[3] * PALM / s
    rot = 0.5 * math.pi - math.atan2(-(k2[1] - k0[1]), k2[0] - k0[0])
    return (cx + ROI_SHIFT * sz * math.sin(rot), cy - ROI_SHIFT * sz * math.cos(rot), ROI_SCALE * sz, rot)


def landmark_crop(bgr, roi):
    cx, cy, size, rot = roi
    sf, c, s = LM / size, math.cos(-rot), math.sin(-rot)
    m = np.array([[sf * c, -sf * s, LM / 2 - sf * (c * cx - s * cy)],
                  [sf * s, sf * c, LM / 2 - sf * (s * cx + c * cy)]])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (cv2.warpAffine(rgb, m, (LM, LM)).astype(np.float32) / 255.0)[None]


class Reader(CalibrationDataReader):
    def __init__(self, name, inputs):
        self.name, self.it = name, iter(inputs)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {self.name: x}


# --------------------------------------------------------------------------- helpers
def session(path, threads):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def timeit(sess, x, n=30):
    name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {name: x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1e3)
    return sorted(t)[n // 2]


def fp32_nodes(path, heads):
    """First conv (raw pixels) and the layers that feed the output decoder stay fp32."""
    g = onnx.load(str(path)).graph
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n.op_type)
    convs = [n for n in g.node if n.op_type == "Conv"]
    keep = [convs[0].name] + [n.name for n in g.node if n.op_type == "Gemm"]
    keep += [n.name for n in convs if users.get(n.output[0]) == ["Transpose"]]  # palm box/score heads
    return [] if heads else keep


def quantize(src, dst, inputs, args, exclude, percentile=False):
    from onnx import version_converter

    model = onnx.load(str(src))
    if model.opset_import[0].version < 13:  # per-channel DequantizeLinear needs opset 13
        src = dst.with_suffix(".opset13.onnx")
        onnx.save(version_converter.convert_version(model, 13), str(src))
    reader = Reader(ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).get_inputs()[0].name, inputs)
    quantize_static(
        str(src), str(dst), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8 if args.act == "u8" else QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile if percentile else CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
        op_types_to_quantize=["Conv"] if args.conv_only else None,
        extra_options={"CalibPercentile": 99.999} if percentile else {},
    )
    if src.name.endswith(".opset13.onnx"):
        src.unlink()


# ------------------------------------------------------------- TF / TFLite -> ONNX
class TFLiteRef:
    """Reference runner for a .tflite file."""

    def __init__(self, path):
        try:
            from ai_edge_litert.interpreter import Interpreter
            self.it = Interpreter(model_path=str(path))
        except ImportError:
            import tensorflow as tf
            self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.ins, self.outs = self.it.get_input_details(), self.it.get_output_details()
        self.input_specs = [(d["name"], tuple(int(s) for s in d["shape"]), d["dtype"]) for d in self.ins]

    def __call__(self, feeds):
        for d, x in zip(self.ins, feeds):
            self.it.set_tensor(d["index"], x)
        self.it.invoke()
        return [self.it.get_tensor(d["index"]).copy() for d in self.outs]


class KerasOrSavedModelRef:
    """Reference runner for a Keras file or SavedModel directory."""

    def __init__(self, path):
        import tensorflow as tf
        self.tf = tf
        if path.is_dir():
            self.model = tf.saved_model.load(str(path))
            fn = self.model.signatures["serving_default"]
            _, kwargs = fn.structured_input_signature
            self.fn, self.keys = fn, list(kwargs)
            self.input_specs = [(k, tuple(kwargs[k].shape.as_list()), kwargs[k].dtype.as_numpy_dtype) for k in self.keys]
        else:
            self.model = tf.keras.models.load_model(str(path), compile=False)
            self.fn = None
            self.input_specs = [(t.name, tuple(t.shape), np.dtype(t.dtype.name)) for t in self.model.inputs]
        self.input_specs = [(n, tuple(1 if s is None else s for s in shp), dt) for n, shp, dt in self.input_specs]

    def __call__(self, feeds):
        if self.fn is not None:
            out = self.fn(**{k: self.tf.constant(x) for k, x in zip(self.keys, feeds)})
            return [np.asarray(v) for v in out.values()]
        out = self.model(feeds if len(feeds) > 1 else feeds[0])
        return [np.asarray(v) for v in (out if isinstance(out, (list, tuple)) else [out])]


def to_onnx(src, dst, opset, nchw):
    try:
        import tf2onnx
    except ImportError as exc:
        sys.exit(f"{exc}\nConversion needs TensorFlow and tf2onnx (Python 3.9-3.12): pip install tensorflow-cpu tf2onnx")
    ref = TFLiteRef(src) if src.suffix == ".tflite" else KerasOrSavedModelRef(src)
    nchw_inputs = [n for n, shp, _ in ref.input_specs if nchw and len(shp) == 4 and shp[-1] in (1, 3)] or None
    if src.suffix == ".tflite":
        tf2onnx.convert.from_tflite(str(src), opset=opset, inputs_as_nchw=nchw_inputs, output_path=str(dst))
    elif src.is_dir():
        tf2onnx.convert.from_saved_model(str(src), opset=opset, inputs_as_nchw=nchw_inputs, output_path=str(dst))
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(src), compile=False)
        spec = [tf.TensorSpec(t.shape, t.dtype, name=t.name.split(":")[0]) for t in model.inputs]
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset, inputs_as_nchw=nchw_inputs,
                                   output_path=str(dst))
    return ref, nchw_inputs


def verify_onnx(ref, dst, nchw_inputs, frames, count, seed, tol):
    import onnx as onnx_lib
    onnx_lib.checker.check_model(onnx_lib.load(str(dst)))
    sess = session(dst, 1)
    onnx_in = sess.get_inputs()
    if len(onnx_in) != len(ref.input_specs):
        raise RuntimeError(f"input count differs: onnx {len(onnx_in)} vs source {len(ref.input_specs)}")
    rng = np.random.default_rng(seed)
    worst, n_out = 0.0, None
    for k in range(count):
        feeds = []
        for name, shp, dt in ref.input_specs:
            if frames and len(shp) == 4 and shp[-1] == 3:  # image input: real frame resized to the model size
                f = cv2.cvtColor(cv2.resize(frames[k % len(frames)], (shp[2], shp[1])), cv2.COLOR_BGR2RGB)
                x = f.astype(np.float32)[None] / 255.0
            elif np.issubdtype(dt, np.floating):
                x = rng.random(shp, dtype=np.float32)
            else:
                x = rng.integers(0, 10, shp).astype(dt)
            feeds.append(x.astype(dt))
        expected = ref(feeds)
        onnx_feeds = {i.name: x.transpose(0, 3, 1, 2) if spec[0] in (nchw_inputs or []) else x
                      for i, x, spec in zip(onnx_in, feeds, ref.input_specs)}
        got = sess.run(None, onnx_feeds)
        if len(got) != len(expected):
            raise RuntimeError(f"output count differs: onnx {len(got)} vs source {len(expected)}")
        for e, g in zip(expected, got):
            if e.shape != g.shape:
                raise RuntimeError(f"output shape differs: onnx {g.shape} vs source {e.shape}")
            worst = max(worst, float(np.max(np.abs(e - g))) / max(1.0, float(np.max(np.abs(e)))))
        n_out = [(g.shape, float(np.max(np.abs(e - g)))) for e, g in zip(expected, got)]
    print(f"    inputs : {[(i.name, i.shape) for i in onnx_in]}")
    print(f"    outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")
    print(f"    max abs diff per output (last run): {[f'{d:.2e}' for _, d in n_out]}; "
          f"worst normalized diff over {count} inputs {worst:.2e} (tol {tol:g})")
    return worst <= tol


def convert_mode(args):
    frames = load_frames(args.source, args.frames, args.seed) if args.source else []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for path in args.convert:
        path = Path(path)
        dst = args.out_dir / (path.stem + ".onnx")
        print(f"{path} -> {dst}")
        try:
            ref, nchw_inputs = to_onnx(path, dst, args.opset, args.nchw)
            ok = verify_onnx(ref, dst, nchw_inputs, frames, args.verify_runs, args.seed, args.tol)
        except Exception as exc:  # keep going so one bad model doesn't hide the others
            print(f"    error: {type(exc).__name__}: {exc}")
            ok = False
        print(f"    {'PASS' if ok else 'FAIL'}  ({dst.stat().st_size / 1e6:.1f} MB)" if dst.exists() else "    FAIL")
        if not ok:
            failed.append(path.name)
    print(f"\n{len(args.convert) - len(failed)}/{len(args.convert)} models converted and verified")
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convert", nargs="+", metavar="MODEL",
                    help="convert .tflite / SavedModel / .keras models to ONNX in --out-dir and verify them, then exit")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset for --convert (default 13)")
    ap.add_argument("--nchw", action="store_true", help="--convert: expose image inputs as NCHW")
    ap.add_argument("--verify-runs", type=int, default=10, help="--convert: inputs compared per model (default 10)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="--convert: max |onnx - source| divided by max(1, max|source|) to pass (default 1e-3)")
    ap.add_argument("--source", nargs="+",
                    help="images, video files and/or a camera index (e.g. 0) with a hand in view")
    ap.add_argument("--frames", type=int, default=150, help="calibration/evaluation frames (default 150)")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--out-dir", type=Path, help="default models/int8, or models/converted with --convert")
    ap.add_argument("--act", choices=["u8", "s8"], default="u8", help="activation type")
    ap.add_argument("--percentile", action="store_true", help="percentile calibration for the landmark net (default min/max)")
    ap.add_argument("--conv-only", action="store_true",
                    help="quantize only Conv nodes; Add/Mul/Relu stay fp32 (helps the decomposed PReLUs)")
    ap.add_argument("--quantize-heads", action="store_true",
                    help="also quantize the first conv and the output heads (default keeps them fp32)")
    ap.add_argument("--threads", type=int, default=1, help="threads for the timing numbers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir = args.out_dir or ROOT / "models" / ("converted" if args.convert else "int8")
    if args.convert:
        convert_mode(args)
    if not args.source:
        ap.error("--source is required unless --convert is given")

    frames = load_frames(args.source, args.frames, args.seed)
    palm32 = session(args.model_dir / "hand_palm.onnx", args.threads)
    lm32 = session(args.model_dir / "hand_landmark_detector.onnx", args.threads)

    palm_in, crops, rois = [], [], []
    for f in frames:
        x, pad = palm_input(f)
        palm_in.append(x)
        roi = best_roi(palm32.run(None, {"input": x})[0], pad)
        rois.append(roi)
        if roi:
            crops.append(landmark_crop(f, roi))
    print(f"{len(frames)} frames, {len(crops)} with a hand found by the fp32 palm net")
    if len(crops) < 10:
        sys.exit("too few frames with a visible hand; use a better --source")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data, pct in (("hand_palm.onnx", palm_in, False), ("hand_landmark_detector.onnx", crops, args.percentile)):
        src = args.model_dir / name
        quantize(src, args.out_dir / name, data, args, fp32_nodes(src, args.quantize_heads), pct)

    palm8 = session(args.out_dir / "hand_palm.onnx", args.threads)
    lm8 = session(args.out_dir / "hand_landmark_detector.onnx", args.threads)

    ds, dc, miss = [], [], 0
    for x, f, roi in zip(palm_in, frames, rois):
        a = palm32.run(None, {"input": x})[0]
        b = palm8.run(None, {"input": x})[0]
        a, b = a[a[:, 0] >= 0.5], b[b[:, 0] >= 0.5]
        if not len(a):
            continue
        if not len(b):
            miss += 1
            continue
        ia, ib = a[np.argmax(a[:, 0])], b[np.argmax(b[:, 0])]
        ds.append(abs(ia[0] - ib[0]))
        dc.append(math.hypot(ia[1] - ib[1], ia[2] - ib[2]) * PALM)
    print(f"\npalm     : hand lost in {miss}/{len(ds) + miss} frames; best-score diff mean {np.mean(ds):.3f}, "
          f"box-centre diff mean {np.mean(dc):.2f}px max {np.max(dc):.2f}px (of 192)")

    err, perr, pres = [], [], []
    for c in crops:
        a = lm32.run(None, {"input_1": c})
        b = lm8.run(None, {"input_1": c})
        w = np.linalg.norm((a[3] - b[3]).reshape(21, 3), axis=1) * 1000
        err.append(w)
        perr.append(np.linalg.norm((a[0] - b[0]).reshape(21, 3)[:, :2], axis=1))
        pres.append(abs(a[1].item() - b[1].item()))
    err, perr = np.array(err), np.array(perr)
    print(f"landmark : world error per point mean {err.mean():.2f}mm, p95 {np.percentile(err, 95):.2f}mm, "
          f"max {err.max():.2f}mm; screen error mean {perr.mean():.2f}px p95 {np.percentile(perr, 95):.2f}px (of 224); "
          f"presence diff max {max(pres):.3f}")

    print(f"\nTiming on this machine, {args.threads} thread(s), median ms (x86 has fast int8, the target may not):")
    xp, xl = palm_in[0], crops[0]
    for label, p, l in (("fp32", palm32, lm32), ("int8", palm8, lm8)):
        print(f"  {label}: palm {timeit(p, xp):7.2f}   landmark {timeit(l, xl):7.2f}")
    sizes = [(m.stat().st_size / 1e6) for m in sorted(args.out_dir.glob("*.onnx"))]
    print(f"\nint8 model sizes: {', '.join(f'{s:.1f} MB' for s in sizes)} written to {args.out_dir}")


if __name__ == "__main__":
    main()
