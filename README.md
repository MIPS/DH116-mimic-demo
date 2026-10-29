# DH116 Hand Mimic

This project tracks a hand with a webcam and the Leadshine DH116EC hand mimics it over EtherCAT. The camera program uses OpenCV and ONNX Runtime; the controller uses SOEM. It does not use ROS or Leadshine's control library.

## Architecture

The camera program (`hand_cam`) and the EtherCAT controller (`mimic_ctrl`) are separate processes. `hand_cam` has a capture thread; inference, preview, and UDP run on its main thread. `mimic_ctrl` receives landmarks over loopback UDP and handles mapping, filtering, and EtherCAT I/O in one 10 ms loop.

```text
camera capture thread -> newest frame -> ONNX inference + preview      (hand_cam)
                                            | UDP landmarks / status
                                            v
                                mimic_ctrl -> SOEM -> EtherCAT hand
```

The UDP messages are defined in [include/mimic_msg.h](include/mimic_msg.h). `hand_cam` sends the 21 world landmarks to `127.0.0.1`; `mimic_ctrl` replies with its status text, the targets, and the measured positions. Each finger joint goes from `0` (open) to `10000` (full stroke). A new target is sent to the hand only when it changes by more than the deadband; in between, the last frame is repeated every cycle.

### What the hand does when ...

| Event | Hand |
|---|---|
| the tracked hand leaves the camera view | holds its last pose |
| `hand_cam` quits (Q/Esc) or `run.sh` gets Ctrl-C | `mimic_ctrl` opens the fingers, then disables the joints |
| the EtherCAT cable is unplugged | `mimic_ctrl` stops after 100 ms without answers; the hand holds its last pose |

Preview keys (in the window, or on the `--web` page): `O` calibrates the open hand, `F` the fist, `R` resets the calibration (deletes `hand_calibration.json`).

## Quick Start

```bash
git clone --recurse-submodules <repository-url>   # or: git submodule update --init
cd DH116-mimic-demo
./scripts/build.sh                                        # first run builds ONNX Runtime; slow
DH116_IFACE=enp0s31f6 ./scripts/run.sh                    # hardware run; see `ip -brief link`
./scripts/run.sh --dry-run                                # camera and inference only; no EtherCAT, no sudo
```


## Build

`./scripts/build.sh` performs these steps and skips any that are already done:

1. Initializes `third_party/SOEM` and `third_party/onnxruntime` if they are empty. ONNX Runtime fetches its own dependencies during its build, so its nested submodules are not initialized.
2. Checks that CMake, a compiler, and OpenCV are available before starting the long build.
3. Builds ONNX Runtime (Release, shared library, no tests) from the submodule into `build/deps/native/ort/` and installs `lib/` and `include/` into `build/prefix/native/`. This is repeated only if the submodule revision changes. On riscv64 it adds the RVV kernels (`--enable_rvv`).
4. Configures and builds `mimic_ctrl` and `hand_cam` into `build/native/`.

To skip the ONNX Runtime compile, point the script at an existing installation:

```bash
./scripts/build.sh --ort-root /path/to/onnxruntime
```

`hand_cam` gets a preview window (needs OpenCV with highgui, `sudo apt install libopencv-dev`). On a machine without a display or without highgui, build it without the window:

```bash
./scripts/build.sh --headless       # hand_cam then always runs as --no-preview
```

### Manual CMake

`scripts/build.sh` wraps standard CMake commands. The equivalent for the controller alone is:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --target mimic_ctrl --parallel
```

Add `-DBUILD_HAND_CAM=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime` for the camera program, and `-DHAND_CAM_HEADLESS=ON` for no preview window. `SOEM_DIR` can select another SOEM source checkout. OpenCV is found through its CMake package configuration. Use a different `-B` directory when changing generators or making an independent configuration.


### Build for RISCV

```bash
./scripts/build-riscv.sh              # toolchain, OpenCV, ONNX Runtime, project, tarball
./scripts/build-riscv.sh --test       # same, then smoke-test under qemu-user (rva23u64, VLEN 256; set QEMU_VLEN=1024 for the FPGA)
```

What the script does:

1. Uses `riscv64-linux-gnu-gcc` from `PATH` (GCC 14 or newer). On Debian and Ubuntu hosts without one, it downloads the cross-compiler packages with `apt-get download` and unpacks them into `build/toolchain/riscv64/`; no root is needed. Elsewhere, install a riscv64 cross GCC yourself.
2. Builds OpenCV 4.10.0 as static libraries: core, imgproc, and videoio with V4L2 capture; no GUI, FFmpeg, or GStreamer. Video files must be MJPEG AVI.
3. Builds ONNX Runtime from the submodule with `--rv64 --enable_rvv` (RVV matrix-multiply, convolution, and softmax kernels) and merges its static libraries into one `libonnxruntime.a`.
4. Builds `mimic_ctrl` and `hand_cam` with [cmake/toolchains/riscv64-rva23-static.cmake](cmake/toolchains/riscv64-rva23-static.cmake), `-DHAND_CAM_HEADLESS=ON`, and static linking, then checks that both are static RISC-V ELF files.
5. Packs `bin/`, `models/`, `hand_calibration.json`, and `run.sh` into the tarball.

The toolchain file spells out the RVA23U64 extensions in `-march` because GCC 15 does not accept `-march=rva23u64`. It deliberately has no `zvl*b` suffix: a `zvl1024b` build lets the compiler assume VLEN >= 1024 and segfaults on shorter-vector cores such as the SpacemiT K3 (VLEN 256), whereas the default build runs on any VLEN. After changing `-march`, rebuild with `--clean-deps`; finished OpenCV and ONNX Runtime steps are otherwise skipped.

Limitations:

- `hand_cam` is headless in this build: it always runs with `--no-preview`, so there is no window. Use `--web PORT` (see [Run](#run)) to see the preview and use the `O`/`F`/`R` calibration keys in a web browser. For the preview window on the board itself, build natively on the board (next section).
- `--test` checks the bundle only under QEMU (the controller in `--dry-run`, and `hand_cam` on a synthetic MJPEG clip). The bundle has also been run on a riscv64 board with a USB camera and the DH116 on EtherCAT.


### Native build on a RISCV board

For the preview window on a monitor attached to the board, build on the board itself with `./scripts/build.sh` (needs `libopencv-dev`, `cmake`, `g++`, and `python3` there). On riscv64, `build.sh` builds ONNX Runtime with `--enable_rvv`; without the RVV kernels it runs 20 to 50 times slower (palm 600 ms instead of 28 ms, under 1 fps). A library built before `build.sh` passed this flag is rebuilt once on the next run, reusing `build/deps/native/ort`.

To check the result:

```bash
grep -E '^(onnxruntime_USE_RVV|HAS_RISCV64_RVV)' build/deps/native/ort/Release/CMakeCache.txt   # ON and 1
nm -C build/prefix/native/lib/libonnxruntime.so | grep -c Rvv     # more than 0
build/native/hand_cam --bench --threads 4 --model-dir models      # palm ~28 ms, landmark ~15 ms
```

If the compiler cannot build the RVV kernels, the ONNX Runtime configure step prints "Falling back to scalar MLAS kernels". ONNX Runtime selects the RVV kernels at run time when the kernel reports the V extension, and `ORT_MLAS_RISCV_FORCE_SCALAR=1` forces the scalar kernels, which compares both with the same library:

```bash
ORT_MLAS_RISCV_FORCE_SCALAR=1 build/native/hand_cam --bench --threads 4 --model-dir models
```




## Run

`run.sh` starts the controller and the camera program and stops both when either exits. It builds first only if an executable is missing.

Camera options such as `--camera 1`, `--hand right`, `--flip-input`, `--no-preview`, and `--web 8080` are passed to `hand_cam`. Give the camera as an index (`--camera 1`), not as `/dev/video1`: a path is opened as a video file.

`--web 8080` also shows the preview in a web browser at `http://<machine-ip>:8080/`, with `O`/`F`/`R` as buttons and as keys on the page. It works in every build, including the headless riscv bundle and over ssh, with or without the preview window. The JPEG encoding runs on its own thread. The server listens on all network interfaces and has no password: anyone who reaches the port can watch the camera and send calibration keys. A browser on the board itself also takes CPU time from inference, so for a monitor on the board the native preview window is lighter.

Controller options can be supplied with `CTRL_ARGS`, for example:
```bash
DH116_IFACE=enp0s31f6 CTRL_ARGS="--current 300 --deadband 80" ./scripts/run.sh
```
`mimic_ctrl` options: `--speed` (default 20000), `--current` (per mille; not sent by default, so the hand uses its own limit), `--deadband`, `--min-cutoff` and `--beta` (filter), `--quiet`. The once-per-second log line lists the targets and measured positions in finger order: thumb sweep, thumb bend, index, middle, ring, little.

The shared UDP port defaults to `5006` and can be changed with `MIMIC_PORT` or `--port`.


### Run on the RISCV Board

The board needs a riscv64 Linux kernel with RVV enabled (`grep isa /proc/cpuinfo` should show `v`) and a V4L2 camera. Copy and unpack the bundle:

```bash
scp dist/dh116-riscv64-rva23.tar.gz user@board:
ssh user@board
tar xzf dh116-riscv64-rva23.tar.gz && cd dh116-riscv64-rva23
```

Then run it with the same `run.sh` options as the native build:

```bash
./run.sh --dry-run --camera 0                          # camera and inference only; no EtherCAT
DH116_IFACE=eth1 ./run.sh --camera 0 --hand left       # hardware run; root, or sudo for EtherCAT
DH116_IFACE=eth1 ./run.sh --camera 0 --web 8080        # same, preview at http://<board-ip>:8080/
```

To start the programs by hand instead:

```bash
./bin/mimic_ctrl --dry-run --calibration hand_calibration.json &
./bin/hand_cam --camera 0 --model-dir models --width 640 --height 480 --threads 4

sudo ./bin/mimic_ctrl eth1 --calibration hand_calibration.json      # hardware; instead of --dry-run
```

`--model-dir` is the directory that holds `hand_palm.onnx` and `hand_landmark_detector.onnx` (default `models`). Lower `--width`, `--height`, or `--det-every` rates if the frame rate is too low, and match `--threads` to the board's core count.

To estimate inference time on any device before building anything, run `python3 scripts/bench_models.py` (needs only `pip install onnxruntime numpy`). It times both networks for several thread counts and prints the expected per-frame model cost; `--threads 1,2,4,8` sweeps counts and `taskset -c ...` pins it to chosen cores.

`scripts/quantize_models.py --source <clip or images or camera index>` (dev host; needs `onnx`, `opencv-python-headless` too) writes int8 versions to `models/int8/` and reports accuracy against fp32. Time them on the device with `bench_models.py --model-dir models/int8`. To time the ONNX Runtime that `hand_cam` is actually linked against (e.g. the RVV build), run `./bin/hand_cam --bench --threads 8 [--model-dir models/int8]`; it needs no camera. This is experimental: with sample images only, the palm net still loses the hand in a fifth of frames and the landmark world error is ~5 mm, so calibrate on a real clip before trusting it.

`--bench` prints the time of each network and a per-frame estimate: `steady` is a frame with the landmark net only, `spike` a frame that also runs the palm net, and `average` spreads the palm net over `--det-every` frames.

`./bin/hand_cam --camera 1 --profile` prints the average time per frame in each stage every 30 frames; use it to see where a slow live run loses its time:

- `wait-camera`: idle time waiting for a frame newer than the last one processed. Capture and MJPEG decoding run on their own thread, so this is not processing time; a large value means the program keeps up with the camera.
- `detect`: from the new frame to the landmarks, including `--flip-input`, the colour conversion, and decoding the network outputs.
- `palm`, `landmark`: the two networks, including the palm input resize and the landmark crop. The palm net runs every `--det-every` frames, and on every frame while no hand is tracked.
- `draw+show`: the overlay, the mirrored preview window and its key handling, and handing the image to `--web`; 0 with `--no-preview` and no `--web`.

Measured on the riscv64 board (SpacemiT X100 cores, `--threads 4`, USB camera with MJPEG 640x480 at 30 fps, its maximum):

| Build | palm | landmark | Live |
|---|---|---|---|
| Static bundle (`build-riscv.sh`, RVV) | 33 ms | 14 ms | 30 fps with a hand (limited by the camera); ~27 fps with no hand, as the palm net then runs on every frame |
| Native, ONNX Runtime without RVV kernels | 602 ms | 812 ms | under 1 fps |
| Native `build.sh` (with `--enable_rvv`) | 28 ms | 14.5 ms | preview shows 33–35 fps; draw+show ~18 ms |

The real frame rate cannot exceed the camera's, because each loop waits for a new frame. The FPS in the preview averages 1/frame-time, so it reads high when frame times vary.



## Repository Map

| Path | Purpose |
|---|---|
| [CMakeLists.txt](CMakeLists.txt) | Top-level targets and dependency discovery. |
| [scripts/build.sh](scripts/build.sh) | One-command build: submodules, ONNX Runtime, and project targets. |
| [scripts/build-riscv.sh](scripts/build-riscv.sh) | Cross-builds the static riscv64 (RVA23) bundle. |
| [cmake/toolchains/](cmake/toolchains/) | CMake toolchain file for the riscv64 build. |
| [scripts/run.sh](scripts/run.sh) | Supervises the camera/controller processes; runs `scripts/build.sh` if needed; a copy also runs inside the riscv64 bundle. |
| [src/hand_cam.cpp](src/hand_cam.cpp) | Camera capture, palm and landmark networks, preview, and UDP to the controller. |
| [src/web_view.h](src/web_view.h) | `hand_cam --web`: the preview as an MJPEG stream for a web browser, and the calibration keys from it. |
| [src/mimic_ctrl.c](src/mimic_ctrl.c) | Controller main loop: UDP from `hand_cam`, landmarks to targets, deadband, log line. |
| [src/dh116.c](src/dh116.c), [include/dh116.h](include/dh116.h) | The hand on EtherCAT (SOEM): start-up, 10 ms cycle, frames sent to the hand, feedback, opening the fingers, shutdown. |
| [src/hand_math.c](src/hand_math.c), [include/hand_math.h](include/hand_math.h) | Landmarks to finger closure, calibration file, One-Euro filter. |
| [include/mimic_msg.h](include/mimic_msg.h) | UDP messages between `hand_cam` and `mimic_ctrl`. |
| [models/](models/) | Palm and hand-landmark ONNX models. |
| [third_party/SOEM/](third_party/SOEM/) | EtherCAT master submodule. |
| [third_party/onnxruntime/](third_party/onnxruntime/) | ONNX Runtime source submodule; built by `build.sh`. |

The top-level CMake project builds `mimic_ctrl`, and `hand_cam` (optional).


## Requirements

- Linux for the current EtherCAT and camera launcher.
- Git, CMake 3.28 or newer, and a C/C++ compiler (`sudo apt install git cmake build-essential`).
- Camera frontend: OpenCV development files (`sudo apt install libopencv-dev`).
- ONNX Runtime source build, only when no existing install is given (see [Build](#build)): Python 3.10 or newer, network access to download ONNX Runtime's own dependencies, and several GB of disk and RAM. Expect a long first build; later runs skip it.
- Hardware use: a DH116EC hand, correctly wired external power, and a dedicated wired Ethernet interface. Verify the exact model's power requirements against its manual.
- `sudo` for EtherCAT access. Stop any other EtherCAT master on the same interface.

The controller needs neither OpenCV nor ONNX Runtime; build it alone with `./scripts/build.sh --controller-only`.