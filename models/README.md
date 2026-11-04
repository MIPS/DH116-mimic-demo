# Models

Hand-tracking networks for `hand_cam` ([`../src/hand_cam.cpp`](../src/hand_cam.cpp)). Two of them run per frame:

1. **Palm detector**: finds hands in the whole frame (letterboxed to a square). Per hand it returns a box plus two keypoints (wrist and middle-finger base), from which `hand_cam` builds a rotated hand crop.
2. **Hand landmark network**: runs on the 224×224 crop of one hand and returns 21 keypoints, a hand-presence score, a left/right score and 21 metric ("world") keypoints.

The Qualcomm folders also hold a third network, a **gesture classifier**, which `hand_cam` does not use.

> **How this was checked.** Names, dtypes and shapes come from the files themselves (ONNX Runtime for `.onnx`, the TFLite flatbuffer for `.tflite`) and from the Qualcomm `metadata.json` files. Value ranges and meanings were confirmed by running every model on images from the FreiHAND evaluation set (ONNX Runtime 1.23 for ONNX, LiteRT for TFLite). The compatibility table at the end was confirmed by running `hand_cam`, built from the current source, on a 60-frame clip. The gesture classifier's input and label description comes from Qualcomm's published model source, not from the files.

## Contents

| Path | What is in it | Format | Weights and I/O |
|---|---|---|---|
| [`palm.onnx`](palm.onnx), [`landmark.onnx`](landmark.onnx) | the pair `hand_cam` loads by default | ONNX | byte-identical copies of `default_FP32/palm.onnx` and `default_FP32/landmark.onnx` |
| [`hand_landmarker.task`](hand_landmarker.task) | MediaPipe Hand Landmarker bundle (zip of `hand_detector.tflite` and `hand_landmarks_detector.tflite`) | `.task` | byte-identical copy of `default_FP32/hand_landmarker.task` |
| [`default_FP32/`](default_FP32) | `palm.onnx` (4.8 MB), `landmark.onnx` (10.9 MB), `hand_landmarker.task` (7.8 MB) | ONNX, `.task` | float32 weights, float32 I/O |
| [`default_INT8/`](default_INT8) | `hand_palm_int8.onnx` (2.0 MB), `hand_landmark_int8.onnx` (3.6 MB) | ONNX | INT8 weights, UINT8 activations (w8a8), float32 I/O |
| [`custom/`](custom) | `palm.onnx` (2.0 MB), `landmark.onnx` (3.6 MB) | ONNX | INT8 weights, UINT16 activations (w8a16), float32 I/O |
| [`Qualcomm_FP32/`](Qualcomm_FP32) | `palm_detector`, `hand_landmark_detector`, `canned_gesture_classifier` (each `.onnx` + `.data`), `metadata.json` | ONNX | float32 weights, float32 I/O |
| [`Qualcomm_W8A8/`](Qualcomm_W8A8) | the same three files, `metadata.json` | ONNX | float32 weights behind Q/DQ nodes (w8a8), **uint8 I/O** |
| [`Google_TFLite_FP16/`](Google_TFLite_FP16) | `palm_detection_lite.tflite` (2.0 MB), `hand_landmark_lite.tflite` (2.1 MB) | TFLite | fp16 weights, float32 I/O |
| [`bench_models.py`](bench_models.py) | ONNX Runtime timing script | Python | copy of `../test/bench_models.py` |

`default_INT8/` and `custom/` were made from the default FP32 models with [`../test/quantize_models.py`](../test/quantize_models.py).

## Conventions

- Batch size is 1. Every image input is a square, **RGB** (not BGR) image.
- Float inputs take pixel values scaled to **0..1** (`pixel / 255`). A −1..1 input breaks the palm models (checked).
- uint8 inputs (Qualcomm W8A8) take the raw 0..255 bytes. Their input quantisation is scale 1/255, zero-point 0, which is the same 0..1 range.
- Quantised tensors decode as `real = (q - zero_point) * scale`.
- Palm input: the frame letterboxed into a square (longer side scaled to the model size, black padding).
- Landmark input: `hand_cam` crops 2.9 × the palm box, shifted half a palm towards the fingers and rotated so the fingers point up.
- Landmarks use MediaPipe's 21-point order (0 = wrist, 9 = middle-finger base, 20 = little-finger tip).

## Palm detector

| Folder → file | Input | Output(s) |
|---|---|---|
| `default_FP32/`, `default_INT8/`, `custom/`, top level → `palm.onnx`, `hand_palm_int8.onnx` | `input` f32 `[1,3,192,192]` (**NCHW**) | `pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y` f32 `[N,8]` |
| `Qualcomm_FP32/` → `palm_detector.onnx` | `image` f32 `[1,3,256,256]` (NCHW) | `box_coords` f32 `[1,2944,18]`, `box_scores` f32 `[1,2944]` |
| `Qualcomm_W8A8/` → `palm_detector.onnx` | `image` **u8** `[1,3,256,256]` (NCHW) | `box_coords` **u8** `[1,2944,18]`, `box_scores` **u8** `[1,2944]` |
| `Google_TFLite_FP16/` → `palm_detection_lite.tflite`; `.task` → `hand_detector.tflite` | `input_1` f32 `[1,192,192,3]` (**NHWC**) | `Identity` f32 `[1,2016,18]`, `Identity_1` f32 `[1,2016,1]` |

The three kinds of output are not interchangeable, so the caller needs different post-processing for each.

**Default family: `[N, 8]` finished detections.**

| Column | Meaning |
|---|---|
| 0 | score, 0..1 (sigmoid already applied) |
| 1, 2 | box centre x, y |
| 3 | box size (side of the square box) |
| 4, 5 | keypoint 0 (wrist) x, y |
| 6, 7 | keypoint 2 (middle-finger base) x, y |

- Coordinates are relative to the 192×192 input (0..1, they can fall slightly outside; −0.19..1.47 seen).
- Anchor decoding, sigmoid and NMS (IoU 0.3, at most 2016 boxes) are inside the graph, but there is **no score threshold**. N is a few hundred and varies per image (299–486 seen). Rows are sorted by score, best first. The caller must drop low scores (`hand_cam` keeps score ≥ 0.5; a clearly visible hand typically scores 0.9–0.98).

**Qualcomm: all 2944 anchors, decoded but not suppressed.**

- `box_coords[0, a]` holds 18 values: `x0, y0, x1, y1`, then 7 keypoints × `(x, y)`. All are in pixels of the 256×256 input (FP32: −67..320 seen). Keypoint 0 is the wrist and keypoint 2 the middle-finger base, the two `hand_cam` uses.
- `box_scores[0, a]` is the score, 0..1 (sigmoid already applied).
- Anchor decoding is inside the graph. There is **no NMS** and no threshold. `hand_cam` keeps score ≥ 0.5 and merges boxes with IoU ≥ 0.3 into one detection by a score-weighted average.
- W8A8: both outputs are uint8; see [Quantisation of the Qualcomm W8A8 models](#quantisation-of-the-qualcomm-w8a8-models).

**Google TFLite: raw network output.**

- `Identity` `[1,2016,18]`: per anchor `dx, dy, w, h`, then 7 keypoints × `(x, y)`, in pixels of the 192×192 input and relative to the anchor. The box centre is `Identity[0, a, 0:2] / 192 + anchor_centre` (checked against the ONNX model).
- `Identity_1` `[1,2016,1]`: raw score **logits** (−32..4 seen). Apply a sigmoid.
- The 2016 anchors are not stored in the model. They are MediaPipe's SSD anchors: a 24×24 grid with 2 anchors per cell plus a 12×12 grid with 6 per cell, fixed size 1.0, centres at `(i + 0.5) / grid`. The caller must build them, decode, apply the sigmoid and run NMS.

## Hand landmark network

| Folder → file | Input | Outputs |
|---|---|---|
| `default_FP32/`, `default_INT8/`, `custom/`, top level → `landmark.onnx`, `hand_landmark_int8.onnx` | `input_1` f32 `[1,224,224,3]` (**NHWC**) | `Identity` `[1,63]`, `Identity_1` `[1,1]`, `Identity_2` `[1,1]`, `Identity_3` `[1,63]`, all f32 |
| `Qualcomm_FP32/` → `hand_landmark_detector.onnx` | `image` f32 `[1,3,224,224]` (**NCHW**) | `landmarks` `[1,63]`, `scores` `[1,1]`, `lr` `[1,1]`, `world_landmarks` `[1,63]`, all f32 |
| `Qualcomm_W8A8/` → `hand_landmark_detector.onnx` | `image` **u8** `[1,3,224,224]` (NCHW) | the same four names, all **u8** |
| `Google_TFLite_FP16/` → `hand_landmark_lite.tflite`; `.task` → `hand_landmarks_detector.tflite` | `input_1` f32 `[1,224,224,3]` (NHWC) | `Identity` … `Identity_3`, same shapes, f32 |

All landmark networks have the same four outputs in the same order. Only the names differ, and `hand_cam` reads them by index.

| # | Default / TFLite name | Qualcomm name | Shape | Meaning |
|---|---|---|---|---|
| 0 | `Identity` | `landmarks` | `[1,63]` | 21 × `(x, y, z)`, flattened. x, y are pixels of the 224×224 crop (they can fall outside it). z is relative depth in the same pixel scale with the wrist at about 0 (−56..42 seen). |
| 1 | `Identity_1` | `scores` | `[1,1]` | hand-presence probability, 0..1 (sigmoid inside the graph). `hand_cam` needs ≥ 0.5. |
| 2 | `Identity_2` | `lr` | `[1,1]` | handedness score, 0..1. `hand_cam` treats > 0.5 as a right hand. |
| 3 | `Identity_3` | `world_landmarks` | `[1,63]` | 21 × `(x, y, z)` in metres, origin near the hand centre (within about ±0.1 m seen). |

## Gesture classifier (Qualcomm only)

`canned_gesture_classifier.onnx` is not used by `hand_cam`.

| | FP32 | W8A8 |
|---|---|---|
| Inputs | `hand` f32 `[1,64]`, `mirrored_hand` f32 `[1,64]` | the same two names, u8 `[1,64]` (scale 0.0131544, zero-point 147, so about −1.93..1.42) |
| Output | `Identity` f32 `[1,8]`, softmax probabilities (they sum to 1) | `Identity` u8 `[1,8]` (scale 1/256, zero-point 0) |

- Each input row is 63 normalised landmark features (21 × 3) plus 1 handedness scalar (0.0 or 1.0). `mirrored_hand` is the same for the x-flipped hand with the handedness inverted. The model files do not say how the features are normalised.
- Class order: `None`, `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.
- Source of the above: Qualcomm AI Hub Models v0.65.0, `qai_hub_models/models/mediapipe_hand_gesture/model.py`.

## Quantisation of the Qualcomm W8A8 models

The values below are in `Qualcomm_W8A8/metadata.json` and match the Q/DQ nodes inside each graph. `hand_cam` hard-codes the palm and landmark ones (`PALM_BOX_Q`, `PALM_SCORE_Q`, `LM_XYZ_Q`, `LM_SCORE_Q`, `LM_WORLD_Q`). If the models are re-exported, update those constants from the new `metadata.json`.

| Model | Tensor | dtype | Scale | Zero-point |
|---|---|---|---|---|
| palm, landmark | `image` (input) | u8 | 0.003921568859 (1/255) | 0 |
| palm | `box_coords` | u8 | 1.6417323350906372 | 50 |
| palm | `box_scores` | u8 | 0.00390625 (1/256) | 0 |
| landmark | `landmarks` | u8 | 1.1388181447982788 | 48 |
| landmark | `scores`, `lr` | u8 | 0.00390625 | 0 |
| landmark | `world_landmarks` | u8 | 0.0009378039976581931 | 122 |
| classifier | `hand`, `mirrored_hand` | u8 | 0.013154414482414722 | 147 |
| classifier | `Identity` | u8 | 0.00390625 | 0 |

## Structural differences

| Aspect | Default family (`default_*`, `custom`, top level) | Qualcomm | Google TFLite |
|---|---|---|---|
| Palm input | 192×192, NCHW | 256×256, NCHW | 192×192, NHWC |
| Landmark input | 224×224, **NHWC** | 224×224, **NCHW** | 224×224, NHWC |
| Palm anchors | 2016 | 2944 | 2016 |
| Palm output | finished detections `[N,8]` | all anchors: boxes `[1,2944,18]` + scores `[1,2944]` | raw regressors `[1,2016,18]` + logits `[1,2016,1]` |
| Done inside the palm graph | anchor decode, sigmoid, NMS | anchor decode, sigmoid | nothing |
| Left to the caller | score threshold | score threshold, NMS (and dequantising for W8A8) | anchors, decode, sigmoid, NMS |
| Landmark output names | `Identity`, `Identity_1..3` | `landmarks`, `scores`, `lr`, `world_landmarks` | `Identity`, `Identity_1..3` |
| Landmark output order and shapes | same in every model | same | same |
| Quantised I/O | float32, even for `default_INT8` and `custom` | uint8 for W8A8 | float32 (fp16 weights) |
| Third model | none | `canned_gesture_classifier` | none |
| Weights | inside the one `.onnx` | external `<name>.data` next to each `.onnx` | inside the one `.tflite` |
| Network size | full palm and landmark (the ones in the `.task`) | palm is a different, 256-px network; landmark is the full one | "lite" variants |
| Graph producer | tf2onnx (palm: IR 8, opset 11; landmark: IR 7, opset 13); quantised: `onnx.quantize` (opset 13; `custom`: opset 21) | Qualcomm AI Hub Workbench (IR 10, opset 21) | TFLite schema v3 |

Worth knowing:

- **Layouts are mixed.** Even in the default pair the palm net is NCHW and the landmark net NHWC. `hand_cam` reads the layout from each model's input shape (`shape[1] == 3` means NCHW), so any mix works.
- **The palm output contract decides the post-processing.** `hand_cam` chooses its palm decoder from the palm *input dtype*: uint8 means Qualcomm-style outputs, anything else means `[N,8]`. That is why `Qualcomm_FP32/palm_detector.onnx` (float input, 18-value output) is rejected.
- **The two quantised ONNX styles differ.**
  - `default_INT8` and `custom` store real INT8 weight tensors (hence the small files) and quantise activations to UINT8 / UINT16 with Q/DQ nodes. They keep float32 inputs and outputs, so calling code does not change. A few output-head layers stay float (palm: 5 of 63 convolutions).
  - `Qualcomm_W8A8` stores the weights as **float32** in `.data` and quantises them with Q→DQ nodes in the graph, so the files are about as large as the FP32 ones (palm 7.5 MB vs 7.8 MB, landmark 11.2 MB vs 11.0 MB). Inputs and outputs are uint8, so the caller must quantise and dequantise.
- **`custom` needs opset 21** (UINT16 Q/DQ), so it needs a recent ONNX Runtime (1.23 loads it). `default_INT8` stays on opset 13.
- **Each Qualcomm `.onnx` loads `<same name>.data` from its own folder.** Keep each pair together under the same names.
- **Lite versus full (TFLite).** `palm_detection_lite` is shallower than the full `hand_detector.tflite` (30 Conv + 23 depthwise layers against 35 + 28; 2.0 MB against 2.3 MB). `hand_landmark_lite` has the same depth as the full `hand_landmarks_detector.tflite` (31 Conv + 16 depthwise) but narrower layers (2.1 MB against 5.5 MB) and a MaxPool the full one lacks. The lite files are a separate, smaller set of weights.

### Same network, different packaging

- `default_FP32/landmark.onnx` is `hand_landmarks_detector.tflite` from the `.task`, converted with tf2onnx. All four outputs agree to 4e-4 over 40 images.
- `default_FP32/palm.onnx` finds the same boxes as `hand_detector.tflite` from the `.task` once the TFLite output is decoded.
- `Qualcomm_FP32/hand_landmark_detector.onnx` is the same network as `default_FP32/landmark.onnx` (47 Conv, 32 Clip, 9 Add, 4 Gemm, 2 Sigmoid in both). Outputs agree to 3e-5; only the input layout (NCHW) and the output names differ.
- The palm detectors are **different networks**: 192 px with 2016 anchors (default, TFLite) against 256 px with 2944 anchors (Qualcomm: 89 Conv + 2 ConvTranspose).
- `default_INT8`, `custom` and `Qualcomm_W8A8` are quantised approximations of their FP32 counterparts, so their outputs differ from the FP32 ones.

## Using the models

```sh
hand_cam --model-dir models                       # default pair (palm.onnx + landmark.onnx)
hand_cam --model-dir models/custom                # a folder holding palm.onnx and landmark.onnx
hand_cam --palm-model models/Qualcomm_W8A8/palm_detector.onnx \
         --landmark-model models/Qualcomm_W8A8/hand_landmark_detector.onnx
python3 models/bench_models.py --model-dir models/Qualcomm_W8A8   # timing only
```

`hand_cam` uses ONNX Runtime, so only `.onnx` files load. `bench_models.py` picks the one `.onnx` whose name contains `palm` and the one containing `landmark`, so the gesture classifier is ignored.

| Folder | `hand_cam` | `bench_models.py --model-dir` |
|---|---|---|
| top level, `default_FP32/` | works (the default) | works |
| `default_INT8/` | works, but pass `--palm-model` and `--landmark-model` (the files are named `hand_*_int8.onnx`) | works |
| `custom/` | works | works |
| `Qualcomm_FP32/` | **palm model fails** with `the palm model output must have 8 values per detection`; the landmark model works (for example with the default palm) | works |
| `Qualcomm_W8A8/` | works with `--palm-model` and `--landmark-model` | works |
| `Google_TFLite_FP16/` | cannot load (`.tflite`) | no `.onnx` found |
