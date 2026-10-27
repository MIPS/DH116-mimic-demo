# Inference results and analysis

Dev board: MilkV Jupiter 2 (SpacemiT K3: 8× X100 up to 2.15 GHz, 8× A100 up to 1.8 GHz).
Measured with `hand_cam --bench` (ORT built from source with RVV, models in `models/`, 100 timed runs, median ms).
Final target: single-core RISC-V FPGA (VLEN 1024, VME/matrix extension), so the 1-thread numbers are the relevant ones.

## Measurements

| Build / model | Threads | Palm | Landmark | Per-frame avg (palm every 30) |
|---|---|---|---|---|
| ORT pip wheel, fp32 | 1 | 324 | 253 | 275 ms (palm every 15) |
| ORT pip wheel, fp32 | 8 | 65 | 50 | 54 ms |
| ORT pip wheel, int8 | 1 | 498 | 367 | 400 ms |
| ORT pip wheel, int8 | 8 | 143 | 72 | 81 ms |
| **ORT source build (RVV), fp32** | 1 | 61.5 | 38.8 | 40.8 ms (~24 fps) |
| **ORT source build (RVV), fp32** | 8 | 27.2 | 8.6 | 9.5 ms (~105 fps) |
| ORT source build (RVV), int8 | 1 | 360 | 219 | 231 ms |
| ORT source build (RVV), int8 | 8 | 117.5 | 39.1 | 43 ms (~23 fps) |

Takeaways:
- The source build is 4–6× faster than the pip wheel in fp32. The K3 already exceeds the ~15 fps target in fp32, even on one thread.
- int8 is about 5.6–5.9× **slower** than fp32 on the same build. Do not use int8 on the K3.
- A segfault in the earlier RISC-V bundle came from `-march=...zvl1024b`: the compiler assumed VLEN ≥ 1024, but the K3 has 256. Fixed (flag removed; `QEMU_VLEN` defaults to 256).

## Why int8 is slower (inferred from ORT source and op counts, not profiled on the board)

1. **No int8 dot-product instruction in plain RVV 1.0.** The MLAS RVV int8 GEMM uses widening multiply-add, which is about as many instructions per MAC as fp32 (~0.7 MAC/cycle int8 vs ~4 MAC/cycle fp32 observed). int8 only wins with an int8 dot/matrix instruction (SpacemiT IME, or your VME). Stock ORT does not use it.
2. **Int8 depthwise conv has no RVV path** (`qdwconv.cpp` is generic C++), and both nets are mostly depthwise + pointwise convs.
3. **Quantize/Dequantize/requantize glue is probably scalar**. The palm net also has `Neg` (from the PReLU decomposition), which has no int8 form, so each of its 31 PReLUs likely leaves a float round trip. Op counts: palm int8 graph has 251 Quantize + 372 Dequantize nodes; landmark has 62 + 154.
4. Q/DQ overhead is not the main cause: palm and landmark slow down by almost the same factor (5.9× vs 5.6×) although palm has far more Q/DQ work.

Still unverified: per-node profile on the board. Cheap checks: ORT `enable_profiling`, and `ORT_MLAS_RISCV_FORCE_SCALAR=1` to see how much RVV is helping.

int8 accuracy (x86, two sample images only): palm missed the hand in ~20% of frames; landmark world-coordinate error 5.5 mm mean, 12.7 mm p95. This needs a real hand-motion clip before int8 is trusted for the FPGA.

Other: thread scaling is weak (fp32 8 threads: 2.3× palm, 4.5× landmark) and p95 is well above the median, probably from threads landing on both X100 and A100 clusters. Next step is `taskset` per cluster.

## Should we use a different runtime (e.g. IREE)?

**K3: no.** fp32 ORT already runs at ~8.5 GMAC/s on one core and far exceeds the frame-rate target. A runtime swap would add effort for no needed gain.

**FPGA: ORT is the wrong tool, but IREE is not obviously the right one either.**
- ORT cannot use a custom VME extension without writing a new MLAS kernel set plus an ORT build for the FPGA (large footprint, needs an OS, malloc, threads).
- IREE compiles the model ahead of time to a small static binary and has a RISC-V LLVM backend with a microkernel (ukernel) mechanism. That is where VME kernels could be plugged in. Costs: its RVV code quality on conv-heavy mobile nets is unverified here, VME would need LLVM intrinsic or ukernel work, and the toolchain is heavy.
- **Our graphs are small and regular:** ~47 convs in landmark (pointwise 1×1, depthwise 3×3, a stem conv, a few Gemm/Add/Clip) and ~63 in palm. A hand-written kernel library (pointwise conv as GEMM, depthwise, stem) with a small generated C graph runner (e.g. generated from the ONNX file) is realistic, gives full control over VLEN 1024 and VME, and has the smallest footprint and latency. That is my recommendation for the FPGA.
- If you want a compiler to do tiling and fusion, spike IREE on landmark only, and compare against ORT fp32 on the K3 first. Also worth a look (not benchmarked here): ncnn and XNNPACK, which have RVV kernels and are much smaller than ORT.

## Lighter models

Not benchmarked; these are options and ideas, ranked by value for the FPGA.

1. **Keep the palm net off the FPGA.** In the host/FPGA split, the host runs palm detection (every N frames) and sends the 224×224 crop; the FPGA only runs landmark every frame. This removes ~50% of the FPGA model cost without any model change.
2. **Distil a smaller landmark student.** We only use six joint features (plus the wrist pose). Train a narrower network (width multiplier ~0.5, input 128–160 instead of 224) to reproduce the current model's output on unlabeled video. Cost scales roughly with input pixels × width², so 160 px at half width is ~8× fewer MACs (~40 MMAC). Needs a training step and a real clip for evaluation.
3. **MediaPipe "lite" variants.** MediaPipe publishes lite versions of both the palm detector and hand landmark model. They are cheaper than the "full" versions, but I haven't verified that our models are the full ones or what the exact savings are. Check the lite variants' ONNX/TFLite conversion first, as it is the least effort.
4. **Smaller palm input or a tiny detector.** Dropping palm input from 192 to 128 costs ~0.44× (needs a re-export with matching anchors and a recheck of recall). A NanoDet- or YOLO-nano-class single-class hand detector is another option but needs a training dataset.
5. **Run palm less often.** Already done via `det_every` plus tracking from the previous landmarks. For single-hand teleop, palm only needs to run on loss of track.

Also noted: fuse the palm net's PReLU (decomposed into Neg/Mul/Relu/Add) into one op to cut ~31 elementwise sets of passes. This helps both ORT and any custom runtime.

## Next steps

1. Per-node profile on the K3 (and `taskset` per cluster).
2. Add `--profile` per-stage timing to `hand_cam`.
3. Record a real hand-motion clip for accuracy comparison (fp32 vs int8 vs smaller models).
4. Decide the FPGA runtime: hand-written kernels vs an IREE spike.
5. Decide palm placement (host vs FPGA).
