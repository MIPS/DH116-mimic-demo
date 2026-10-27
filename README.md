# DH116 Hand Teleoperation

This project tracks a hand with a webcam and streams position targets to a Leadshine DH116EC hand over EtherCAT. The camera frontend uses OpenCV and ONNX Runtime; the controller uses SOEM. It does not use ROS or Leadshine's control library.

## Architecture

The camera frontend and EtherCAT controller are separate processes. The camera process has a capture worker thread; inference, preview, and UDP communication run on its main thread. The C controller receives landmarks over loopback UDP and handles mapping, filtering, and EtherCAT I/O in one 5 ms loop.

```text
camera capture thread -> newest frame -> ONNX inference + GUI
                                            | UDP landmarks/status
                                            v
                              dh116_teleop -> SOEM -> EtherCAT hand
```

The UDP packet layout is shared in [include/teleop_proto.h](include/teleop_proto.h). The camera sends 21 world landmarks to `127.0.0.1`; the controller replies with tracking state, target positions, and measured positions. The controller maps closure values to raw targets from `0` (open) to `10000` (full stroke).

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
3. Builds ONNX Runtime (Release, shared library, no tests) from the submodule into `build/deps/native/ort/` and installs `lib/` and `include/` into `build/prefix/native/`. This is repeated only if the submodule revision changes.
4. Configures and builds `dh116_teleop` and `hand_cam` into `build/native/`.

To skip the ONNX Runtime compile, point the script at an existing installation:

```bash
./scripts/build.sh --ort-root /path/to/onnxruntime
```

### Manual CMake

`scripts/build.sh` wraps standard CMake commands. The equivalent for the controller alone is:

```bash
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --target dh116_teleop --parallel
```

Add `-DBUILD_HAND_CAM=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime` for the camera frontend. `SOEM_DIR` can select another SOEM source checkout. OpenCV is found through its CMake package configuration. Use a different `-B` directory when changing generators or making an independent configuration.


### Build for RISCV

```bash
./scripts/build-riscv.sh              # toolchain, OpenCV, ONNX Runtime, project, tarball
./scripts/build-riscv.sh --test       # same, then smoke-test under qemu-user (rva23u64, VLEN 256; set QEMU_VLEN=1024 for the FPGA)
```

What the script does:

1. Uses `riscv64-linux-gnu-gcc` from `PATH` (GCC 14 or newer). On Debian and Ubuntu hosts without one, it downloads the cross-compiler packages with `apt-get download` and unpacks them into `build/toolchain/riscv64/`; no root is needed. Elsewhere, install a riscv64 cross GCC yourself.
2. Builds OpenCV 4.10.0 as static libraries: core, imgproc, and videoio with V4L2 capture; no GUI, FFmpeg, or GStreamer. Video files must be MJPEG AVI.
3. Builds ONNX Runtime from the submodule with `--rv64 --enable_rvv` (RVV matrix-multiply, convolution, and softmax kernels) and merges its static libraries into one `libonnxruntime.a`.
4. Builds `dh116_teleop` and `hand_cam` with [cmake/toolchains/riscv64-rva23-static.cmake](cmake/toolchains/riscv64-rva23-static.cmake), `-DHAND_CAM_HEADLESS=ON`, and static linking, then checks that both are static RISC-V ELF files.
5. Packs `bin/`, `models/`, `hand_calibration.json`, and `run.sh` into the tarball.

The toolchain file spells out the RVA23U64 extensions in `-march` because GCC 15 does not accept `-march=rva23u64`. It deliberately has no `zvl*b` suffix: a `zvl1024b` build lets the compiler assume VLEN >= 1024 and segfaults on shorter-vector cores such as the SpacemiT K3 (VLEN 256), whereas the default build runs on any VLEN. After changing `-march`, rebuild with `--clean-deps`; finished OpenCV and ONNX Runtime steps are otherwise skipped.

Limitations:

- `hand_cam` is headless in this build: it always runs with `--no-preview`, so there is no window and no `o`/`f`/`r` calibration keys. Calibrate with the native build and copy `hand_calibration.json` into the bundle before running.
- Verified only under QEMU emulation (the controller in `--dry-run`, and `hand_cam` on a synthetic MJPEG clip). Not verified on real RISC-V hardware, with a real camera, or against a DH116 on EtherCAT.




## Run

`run.sh` starts the controller and the camera frontend and stops both when either exits. It builds first only if an executable is missing. 

Camera options such as `--camera 1`, `--hand right`, `--flip-input`, and `--no-preview` are passed to `hand_cam`. 
Controller options can be supplied with `TELEOP_ARGS`, for example:
```bash
DH116_IFACE=enp0s31f6 TELEOP_ARGS="--cur 600 --deadband 80" ./scripts/run.sh
```
The shared UDP port defaults to `5006` and can be changed with `TELEOP_PORT` or `--port`.


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
```

To start the programs by hand instead:

```bash
./bin/dh116_teleop --dry-run --calibration hand_calibration.json &
./bin/hand_cam --camera 0 --model-dir models --width 640 --height 480 --threads 4

sudo ./bin/dh116_teleop eth1 --calibration hand_calibration.json      # hardware; instead of --dry-run
```

`--model-dir` is the directory that holds `hand_palm.onnx` and `hand_landmark_detector.onnx` (default `models`). Lower `--width`, `--height`, or `--det-every` rates if the frame rate is too low, and match `--threads` to the board's core count.

To estimate inference time on any device before building anything, run `python3 scripts/bench_models.py` (needs only `pip install onnxruntime numpy`). It times both networks for several thread counts and prints the expected per-frame model cost; `--threads 1,2,4,8` sweeps counts and `taskset -c ...` pins it to chosen cores.

`scripts/quantize_models.py --source <clip or images or camera index>` (dev host; needs `onnx`, `opencv-python-headless` too) writes int8 versions to `models/int8/` and reports accuracy against fp32. Time them on the device with `bench_models.py --model-dir models/int8`. To time the ONNX Runtime that `hand_cam` is actually linked against (e.g. the RVV build), run `./bin/hand_cam --bench --threads 8 [--model-dir models/int8]`; it needs no camera. This is experimental: with sample images only, the palm net still loses the hand in a fifth of frames and the landmark world error is ~5 mm, so calibrate on a real clip before trusting it.

`./bin/hand_cam --camera 1 --profile` prints the average time per frame in each stage (camera wait, palm, landmark, draw+show) every 30 frames; use it to see where a slow live run loses its time.



## Repository Map

| Path | Purpose |
|---|---|
| [CMakeLists.txt](CMakeLists.txt) | Top-level targets and dependency discovery. |
| [scripts/build.sh](scripts/build.sh) | One-command build: submodules, ONNX Runtime, and project targets. |
| [scripts/build-riscv.sh](scripts/build-riscv.sh) | Cross-builds the static riscv64 (RVA23) bundle. |
| [cmake/toolchains/](cmake/toolchains/) | CMake toolchain file for the riscv64 build. |
| [scripts/run.sh](scripts/run.sh) | Supervises the camera/controller processes; runs `scripts/build.sh` if needed; a copy also runs inside the riscv64 bundle. |
| [src/hand_cam.cpp](src/hand_cam.cpp) | Camera capture, ONNX inference, preview, and UDP client. |
| [src/dh116_teleop.c](src/dh116_teleop.c) | UDP receiver, hand mapping, target streaming, and controller loop. |
| [src/dh116_bus.c](src/dh116_bus.c) | SOEM initialization, 5 ms process-data cycle, feedback, and shutdown. |
| [src/hand_math.c](src/hand_math.c) | Landmark features, calibration, and filtering. |
| [include/dh116_pdo.h](include/dh116_pdo.h) | EtherCAT payload and feedback encoders. |
| [include/teleop_proto.h](include/teleop_proto.h) | Camera/controller UDP protocol. |
| [models/](models/) | Palm and hand-landmark ONNX models. |
| [third_party/SOEM/](third_party/SOEM/) | EtherCAT master submodule. |
| [third_party/onnxruntime/](third_party/onnxruntime/) | ONNX Runtime source submodule; built by `build.sh`. |

The top-level CMake project builds `dh116_teleop`, and `hand_cam` (optional).


## Requirements

- Linux for the current EtherCAT and camera launcher.
- Git, CMake 3.28 or newer, and a C/C++ compiler (`sudo apt install git cmake build-essential`).
- Camera frontend: OpenCV development files (`sudo apt install libopencv-dev`).
- ONNX Runtime source build, only when no existing install is given (see [Build](#build)): Python 3.10 or newer, network access to download ONNX Runtime's own dependencies, and several GB of disk and RAM. Expect a long first build; later runs skip it.
- Hardware use: a DH116EC hand, correctly wired external power, and a dedicated wired Ethernet interface. Verify the exact model's power requirements against its manual.
- `sudo` for EtherCAT access. Stop any other EtherCAT master on the same interface.

The controller needs neither OpenCV nor ONNX Runtime; build it alone with `./scripts/build.sh --controller-only`.