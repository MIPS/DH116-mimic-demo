"""Webcam + MediaPipe front end for the physical DH116 hand (scripts/teleop/dh116_teleop).

Per camera frame: grab the newest frame, run MediaPipe's Hand Landmarker, send the 21 world
landmarks of the chosen hand to dh116_teleop over UDP (127.0.0.1, layout in teleop_proto.h) and
draw the reply: skeleton, the commanded closure per joint (orange bar) and the measured joint
position (green tick). All conversion math lives in C; this script only sees and draws.

    ./run.sh [--dry-run] [--hand right] [--camera 1] ...        (starts both programs)

Keys: O open hand   F fist   R reset calibration   Q / Esc quit (the robot hand opens, then disables)
Everything runs locally; frames never leave the machine.
"""

import argparse
from pathlib import Path
import socket
import struct
import sys
import threading
import time

import cv2
import math
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parents[2] / "models" / "hand_landmarker.task"
# ONNX models derived from the same directory as the .task file.
# See README.md → "ONNX model setup" for extraction / conversion steps.
PALM_ONNX = MODEL_PATH.parent / "hand_palm.onnx"
LM_ONNX   = MODEL_PATH.parent / "hand_landmark_detector.onnx"

# teleop_proto.h
DEFAULT_PORT = 5006
VERSION = 1
MAGIC_LANDMARKS = 0x4D4C4844
MAGIC_STATUS = 0x54534844
LANDMARKS = struct.Struct("<IHBBIB3xd63f4x")
STATUS = struct.Struct("<IHBBI6f6h6hB3x64s")
FLAG_DETECTED, FLAG_QUIT = 0x01, 0x02
CAL_NONE, CAL_OPEN, CAL_FIST, CAL_RESET = 0, 1, 2, 3
STATES = {0: "DRY RUN (no EtherCAT)", 1: "hand RUNNING", 2: "stopping"}

LABELS = ("Thumb sweep", "Thumb bend", "Index", "Middle", "Ring", "Little")
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]
HELP = "O: open hand   F: fist   R: reset   Q/Esc: quit"
WINDOW = "DH116 Teleop"
REPLY_TIMEOUT = 0.5  # seconds without a reply before the GUI reports the controller as silent


# ── Minimal stubs matching the mediapipe landmark / handedness API ──────────────

class _LM:
    """Landmark with .x .y .z (screen: normalised [0,1]; world: metres)."""
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z=0.0): self.x, self.y, self.z = float(x), float(y), float(z)

class _Cat:
    """Handedness category with .category_name and .score."""
    __slots__ = ("category_name", "score")
    def __init__(self, name, score): self.category_name, self.score = name, float(score)

class _DetResult:
    __slots__ = ("hand_landmarks", "hand_world_landmarks", "handedness")
    def __init__(self, hl, hwl, hd):
        self.hand_landmarks, self.hand_world_landmarks, self.handedness = hl, hwl, hd


class HandLandmarker:
    """ONNX-Runtime drop-in for mp.tasks.vision.HandLandmarker (VIDEO mode).

    Requires two ONNX models in the models/ directory:
      hand_palm.onnx     – palm detector with post-processing baked in
                           Source: PINTO0309/PINTO_model_zoo 033_Hand_Detection_and_Tracking
                           File:   palm_detection_full_inf_post_192x192.onnx
                           Rename to hand_palm.onnx

      hand_landmark.onnx – hand landmark regressor (4 outputs: screen lms, presence,
                           handedness, world lms)
                           Source: convert hand_landmarks_detector.tflite extracted
                           from hand_landmarker.task (unzip hand_landmarker.task)
                           Command: python -m tf2onnx.convert \\
                               --tflite hand_landmarks_detector.tflite \\
                               --output hand_landmark.onnx --opset 13
    """

    _PALM_SZ   = 192      # palm detection model input side
    _LM_SZ     = 224      # landmark model input side
    _ROI_SCALE = 2.9      # box → ROI expansion (MediaPipe palm_detection_post.cc)
    _ROI_SHIFT = 0.5      # fraction of box_size to shift ROI center toward fingers

    def __init__(self, palm_path, lm_path, num_hands: int = 2, thresh: float = 0.5,
                 threads: int = 4, det_every: int = 15):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads   # tuned for RISC-V (was 16)
        opts.inter_op_num_threads = 1
        self._palm     = ort.InferenceSession(str(palm_path), sess_options=opts)
        self._lm       = ort.InferenceSession(str(lm_path),   sess_options=opts)
        self._num      = num_hands
        self._thresh   = thresh
        self._palm_in  = self._palm.get_inputs()[0].name
        self._lm_in    = self._lm.get_inputs()[0].name
        lm_in_shape    = self._lm.get_inputs()[0].shape
        self._lm_nchw  = len(lm_in_shape) == 4 and lm_in_shape[1] == 3
        self._has_world = len(self._lm.get_outputs()) >= 4
        # Pre-allocated palm letterbox buffer (avoids np.zeros each frame)
        S = self._PALM_SZ
        self._padded = np.zeros((S, S, 3), dtype=np.uint8)
        # ROI tracking: list of (roi_cx, roi_cy, roi_sz, rotation) per hand.
        # Updated each frame from landmark positions; reset on palm re-detect.
        # Palm detection runs every det_every frames (1 = disable tracking).
        self._tracked     = []
        self._det_every   = det_every
        self._det_counter = 0

    def _rois_from_palm_dets(self, dets, pad_x, pad_y, scale):
        S = self._PALM_SZ
        rois = []
        for det in dets:
            cx, cy, sz = float(det[1]), float(det[2]), float(det[3])
            k0x, k0y   = float(det[4]), float(det[5])
            k2x, k2y   = float(det[6]), float(det[7])
            cx_px  = (cx  * S - pad_x) / scale
            cy_px  = (cy  * S - pad_y) / scale
            k0x_px = (k0x * S - pad_x) / scale
            k0y_px = (k0y * S - pad_y) / scale
            k2x_px = (k2x * S - pad_x) / scale
            k2y_px = (k2y * S - pad_y) / scale
            sz_px  = sz * S / scale
            rotation = 0.5 * math.pi - math.atan2(-(k2y_px - k0y_px), k2x_px - k0x_px)
            roi_sz   = self._ROI_SCALE * sz_px
            roi_cx   = cx_px + self._ROI_SHIFT * sz_px * math.sin(rotation)
            roi_cy   = cy_px - self._ROI_SHIFT * sz_px * math.cos(rotation)
            rois.append((roi_cx, roi_cy, roi_sz, rotation))
        return rois

    @staticmethod
    def _roi_from_lms(xy_orig):
        """Estimate next-frame ROI from back-projected landmark pixels [21, 2]."""
        dv = xy_orig[9] - xy_orig[0]   # wrist → middle MCP
        rotation = 0.5 * math.pi - math.atan2(-float(dv[1]), float(dv[0]))
        bb_min = xy_orig.min(axis=0)
        bb_max = xy_orig.max(axis=0)
        roi_cx = float(bb_min[0] + bb_max[0]) * 0.5
        roi_cy = float(bb_min[1] + bb_max[1]) * 0.5
        roi_sz = float(np.linalg.norm(bb_max - bb_min)) * 1.2  # diagonal + 20 % margin
        return roi_cx, roi_cy, roi_sz, rotation

    def detect(self, rgb: np.ndarray) -> _DetResult:
        """Run full hand-landmark pipeline on an RGB uint8 frame."""
        h, w = rgb.shape[:2]

        self._det_counter += 1
        run_palm = not self._tracked or (self._det_counter % self._det_every == 0)

        if run_palm:
            # ── 1. Letterbox to S×S ──────────────────────────────────────────
            S     = self._PALM_SZ
            scale = S / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            pad_y  = (S - nh) // 2
            pad_x  = (S - nw) // 2
            self._padded.fill(0)
            self._padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(rgb, (nw, nh))

            # ── 2. Palm detection (post-processing baked into ONNX model) ────
            palm_in  = (self._padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
            palm_raw = self._palm.run(None, {self._palm_in: palm_in})[0]
            # Expected layout per row: [score, cx, cy, box_size, kp0_x, kp0_y, kp2_x, kp2_y]
            boxes = palm_raw.reshape(-1, palm_raw.shape[-1])
            above = boxes[boxes[:, 0] >= self._thresh]
            dets  = above[np.argsort(above[:, 0])[::-1]][: self._num]
            self._tracked = self._rois_from_palm_dets(dets, pad_x, pad_y, scale)

        hand_lms_list, world_lms_list, handedness_list = [], [], []
        next_tracked = []

        for roi_cx, roi_cy, roi_sz, rotation in self._tracked:
            # ── 3/4. Affine warp: rotated ROI → L×L crop ─────────────────────
            L  = self._LM_SZ
            sf = L / roi_sz                        # pixels-per-pixel scale factor
            cr, sr = math.cos(-rotation), math.sin(-rotation)
            M  = np.float32([
                [sf * cr, -sf * sr, L / 2 - sf * (cr * roi_cx - sr * roi_cy)],
                [sf * sr,  sf * cr, L / 2 - sf * (sr * roi_cx + cr * roi_cy)],
            ])
            crop = cv2.warpAffine(rgb, M, (L, L))

            # ── 5. Landmark inference ─────────────────────────────────────────
            lm_in = crop.astype(np.float32) / 255.0
            lm_in = lm_in.transpose(2, 0, 1)[np.newaxis] if self._lm_nchw else lm_in[np.newaxis]
            lm_out = self._lm.run(None, {self._lm_in: lm_in})
            # outputs: [screen_lms [1,63], presence [1,1], handedness [1,1], world_lms [1,63]]

            if float(lm_out[1].flat[0]) < self._thresh:
                # Tracking lost – drop this hand; palm detector will recover next cycle
                continue

            raw_hand = float(lm_out[2].flat[0])
            side     = "Right" if raw_hand > 0.5 else "Left"
            cat_sc   = raw_hand if raw_hand > 0.5 else 1.0 - raw_hand

            # ── 6. Back-project screen landmarks to original image ────────────
            pts_crop  = lm_out[0].reshape(21, 3)
            M_inv     = cv2.invertAffineTransform(M)
            xy_h      = np.hstack([pts_crop[:, :2], np.ones((21, 1), dtype=np.float32)])
            xy_orig   = (M_inv @ xy_h.T).T        # [21, 2] original image pixels
            screen    = [_LM(xy_orig[i, 0] / w, xy_orig[i, 1] / h, pts_crop[i, 2] / L)
                         for i in range(21)]

            # ── 7. World landmarks (metres, model output[3] when available) ───
            if self._has_world:
                wld   = lm_out[3].reshape(21, 3)
                world = [_LM(wld[i, 0], wld[i, 1], wld[i, 2]) for i in range(21)]
            else:
                # Approximate from screen landmarks when world output is absent
                world = [_LM(screen[i].x - screen[0].x,
                             screen[i].y - screen[0].y, 0.0) for i in range(21)]

            hand_lms_list.append(screen)
            world_lms_list.append(world)
            handedness_list.append([_Cat(side, cat_sc)])

            # ── 8. Update tracking ROI from current landmark positions ─────────
            next_tracked.append(self._roi_from_lms(xy_orig))

        # If all hands were lost, clear so palm detector runs on the next frame
        self._tracked = next_tracked if next_tracked else []

        return _DetResult(hand_lms_list, world_lms_list, handedness_list)

    def close(self):  # API compatibility with mediapipe
        pass


def encode(seq: int, time_s: float, right: bool, world=None, cal: int = CAL_NONE, quit_: bool = False) -> bytes:
    flags = (FLAG_DETECTED if world is not None else 0) | (FLAG_QUIT if quit_ else 0)
    coords = [c for point in world for c in point] if world is not None else [0.0] * 63
    return LANDMARKS.pack(MAGIC_LANDMARKS, VERSION, flags, cal, seq, int(right), time_s, *coords)


def decode_status(payload: bytes):
    if len(payload) != STATUS.size:
        return None
    v = STATUS.unpack(payload)
    if v[0] != MAGIC_STATUS or v[1] != VERSION:
        return None
    return {"state": v[2], "valid": v[3], "seq": v[4], "closure": v[5:11], "target": v[11:17],
            "measured": v[17:23], "have_fb": v[23], "msg": v[24].split(b"\0", 1)[0].decode(errors="replace")}


class Grabber(threading.Thread):
    """Reads the camera continuously and keeps only the newest frame, so a slow inference step
    never processes a stale, buffered frame (lowest latency) and capture overlaps inference."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        super().__init__(daemon=True)
        self.capture, self.cond = capture, threading.Condition()
        self.frame, self.stamp, self.count, self.failures, self.running = None, 0.0, 0, 0, True

    def run(self) -> None:
        while self.running:
            ok, image = self.capture.read()
            now = time.monotonic()
            with self.cond:
                if ok:
                    self.frame, self.stamp, self.count, self.failures = image, now, self.count + 1, 0
                else:
                    self.failures += 1
                self.cond.notify_all()
            if not ok:
                time.sleep(0.01)

    def next(self, last_count: int, timeout: float = 1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.count != last_count or self.failures > 30, timeout)
            if self.failures > 30:
                raise RuntimeError("The camera stopped delivering frames.")
            return self.frame, self.stamp, self.count


def open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    if source.isdigit():
        capture = cv2.VideoCapture(int(source), cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        # no CAP_PROP_BUFFERSIZE=1: it drops this webcam to 20 FPS; Grabber reads continuously anyway
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera or video: {source}")
    return capture


def draw_preview(image: np.ndarray, hands: list, selected, status, lines: list) -> np.ndarray:
    height, width = image.shape[:2]
    for index, landmarks in enumerate(hands):
        points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        color = (80, 220, 80) if index == selected else (150, 150, 150)
        for start, end in CONNECTIONS:
            cv2.line(image, points[start], points[end], color, 2)
        for point in points:
            cv2.circle(image, point, 3, (40, 40, 230) if index == selected else color, -1)
    image = cv2.flip(image, 1)  # selfie view
    # Opaque background strips – avoids image.copy() + addWeighted (saves ~2 ms at 320×240)
    cv2.rectangle(image, (0, 0), (width, 14 + 20 * len(lines)), (0, 0, 0), -1)
    cv2.rectangle(image, (0, height - 30 - 20 * len(LABELS)), (272, height), (0, 0, 0), -1)
    for row, line in enumerate(lines):
        cv2.putText(image, line, (10, 22 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for row, label in enumerate(LABELS):
        y = height - 12 - 20 * (len(LABELS) - 1 - row)
        cv2.putText(image, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(image, (110, y - 11), (260, y + 1), (60, 60, 60), -1)
        if status is not None:
            cv2.rectangle(image, (110, y - 11), (110 + int(150 * status["target"][row] / 10000), y + 1), (80, 200, 255), -1)
            if status["have_fb"]:
                x = 110 + int(150 * min(max(status["measured"][row], 0), 10000) / 10000)
                cv2.line(image, (x, y - 14), (x, y + 4), (80, 255, 80), 2)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="0", help="Webcam index or video file (default: %(default)s)")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--hand", choices=("left", "right"), default="left", help="Your hand to track; right is mirrored onto the left robot hand")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="dh116_teleop UDP port (default: %(default)s)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH,
                        help="Path used to derive hand_palm.onnx and hand_landmark.onnx in the same directory")
    parser.add_argument("--flip-input", action="store_true", help="Un-mirror a camera that delivers selfie-mirrored frames")
    parser.add_argument("--no-preview", action="store_true", help="No window (stop with Ctrl+C)")
    parser.add_argument("--frames", type=int, default=0, help="Stop after this many frames; 0 = unlimited")
    parser.add_argument("--threads", type=int, default=8,
                        help="ONNX intra-op thread count; tune to your core count (default: %(default)s)")
    parser.add_argument("--det-every", type=int, default=30,
                        help="Re-run palm detector every N frames; 1 = disable ROI tracking (default: %(default)s)")
    args = parser.parse_args()
    if args.frames < 0 or not 0 < args.port <= 65535:
        parser.error("--frames must be nonnegative and --port 1-65535")
    palm_onnx = args.model.parent / "hand_palm.onnx"
    lm_onnx   = args.model.parent / "hand_landmark_detector.onnx"
    for p in (palm_onnx, lm_onnx):
        if not p.exists():
            parser.exit(2, f"ONNX model not found: {p}\n"
                           "See the HandLandmarker docstring in hand_cam.py for conversion steps.\n")
    try:
        landmarker = HandLandmarker(palm_onnx, lm_onnx, num_hands=1,
                                    threads=args.threads, det_every=args.det_every)
        capture = open_camera(args.camera, args.width, args.height)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"hand_cam error: {error}\n")

    is_file = not args.camera.isdigit()
    grabber = None if is_file else Grabber(capture)
    if grabber:
        grabber.start()
    target = ("127.0.0.1", args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    right = args.hand == "right"
    keys = {ord("o"): CAL_OPEN, ord("f"): CAL_FIST, ord("r"): CAL_RESET}
    seq, frames, count, cal, last_timestamp, fps = 0, 0, 0, CAL_NONE, -1, 0.0  # seq 0 tells the controller "new GUI"
    status, status_time, message = None, 0.0, ""
    previous = time.monotonic()
    print(f"Tracking your {args.hand} hand -> dh116_teleop on 127.0.0.1:{args.port}. {HELP}", flush=True)
    try:
        while not args.frames or frames < args.frames:
            if grabber:
                image, stamp, count = grabber.next(count)
                if image is None:
                    continue
            else:
                ok, image = capture.read()
                stamp = time.monotonic()
                if not ok:
                    break
            if args.flip_input:
                image = cv2.flip(image, 1)
            last_timestamp = max(last_timestamp + 1, int(stamp * 1000))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(rgb)
            matches = [(hand[0].score, index) for index, hand in enumerate(result.handedness)
                       if hand[0].category_name.lower() == args.hand]
            selected = max(matches)[1] if matches else None
            world = None if selected is None else [(p.x, p.y, p.z) for p in result.hand_world_landmarks[selected]]
            try:
                sock.sendto(encode(seq, stamp, right, world, cal), target)
            except OSError as error:
                message = f"Send failed: {error}"
            cal = CAL_NONE
            # never wait for the controller: take whatever replies are queued (bars may be one frame old)
            while True:
                try:
                    reply = decode_status(sock.recv(256))
                except (BlockingIOError, OSError):
                    break
                if reply:
                    status, status_time = reply, time.monotonic()
            seq += 1
            frames += 1
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - previous, 1e-3) if frames > 1 else 0.0
            previous = now
            alive = status is not None and now - status_time < REPLY_TIMEOUT
            if alive and status["msg"] != message:
                message = status["msg"]
                print(message, flush=True)
            if args.no_preview:
                continue
            lines = [f"{args.hand.upper()} hand   {fps:4.1f} FPS   " + ("Hand detected" if world else f"No {args.hand} hand visible"),
                     STATES.get(status["state"], "?") if alive else f"No reply from dh116_teleop on port {args.port}",
                     message, HELP]
            cv2.imshow(WINDOW, draw_preview(image, result.hand_landmarks, selected, status if alive else None, lines))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            cal = keys.get(key, CAL_NONE)
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print(f"hand_cam error: {error}", file=sys.stderr)
        return 2
    finally:
        for _ in range(3):  # tell the controller to open the hand and stop (UDP, so repeat)
            try:
                sock.sendto(encode(seq, time.monotonic(), right, quit_=True), target)
            except OSError:
                pass
            seq += 1
        sock.close()
        if grabber:
            grabber.running = False
            grabber.join(timeout=1.0)
        capture.release()
        landmarker.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
