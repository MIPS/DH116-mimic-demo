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
| a joint reports alarm stop, emergency stop, or is no longer enabled | `mimic_ctrl` prints the joint in red, disables all joints without moving them, and exits with code 3 |
| a joint reports a travel limit, homing, or torque reached (at its current limit, e.g. a finger pushing on something) for 100 ms | `mimic_ctrl` prints a yellow warning, and a line when it clears, and keeps going |

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

`run.sh` starts the controller and the camera program and stops both when either exits. It builds first only if an executable is missing. `./scripts/run.sh --help` prints the same cheat-sheet:

| I want to | Command |
|---|---|
| test camera and inference, no robot | `./scripts/run.sh --dry-run` |
| drive the hand | `DH116_IFACE=enp0s31f6 ./scripts/run.sh` |
| use another camera or UDP port | `--camera 1`, `--port 5007` |
| lower the camera frame rate | `--fps 15` (default 30; the camera driver may ignore it) |
| use other models | `--palm-model FILE --landmark-model FILE` (input size is read from the model), or `--model-dir DIR` for `palm.onnx` and `landmark.onnx` |
| change speed, current, deadband | `CTRL_ARGS="--current 300 --deadband 80"` (see below) |
| see the preview in a browser | `--web 8080` |
| one image per button press, not real time | `--web 8080 --snapshot` |
| the same, with the picture from a laptop's webcam instead of a camera on the board | `--camera browser --web 8008 --snapshot`, with an ssh tunnel (see below) |

Options that `run.sh` does not know are passed to `hand_cam`. `run.sh` itself starts it with `--width 640 --height 480 --threads 4 --det-every 30 --model-dir models`; repeat any of them to override (the last value wins).

Camera options such as `--camera 1`, `--hand right`, `--flip-input`, `--no-preview`, and `--web 8080` are passed to `hand_cam`. Give the camera as an index (`--camera 1`), not as `/dev/video1`: a path is opened as a video file.

`--web 8080` also shows the preview in a web browser at `http://<machine-ip>:8080/`, with `O`/`F`/`R` as buttons and as keys on the page. It works in every build, including the headless riscv bundle and over ssh, with or without the preview window. The JPEG encoding runs on its own thread. The server listens on all network interfaces and has no password: anyone who reaches the port can watch the camera and send calibration keys. A browser on the board itself also takes CPU time from inference, so for a monitor on the board the native preview window is lighter.

`--web 8080 --snapshot` is the not-real-time mode (`--snapshot` needs `--web`). The page shows the live camera next to the last capture, with a **Capture** button. One press takes the newest camera frame and runs the palm detector and the landmark net on it once; the status line flashes "Inferencing..." until they are done. If no hand of the `--hand` side is found, the page says so and nothing is sent. Otherwise the landmarks are drawn on the captured image, and one command goes to `mimic_ctrl` (after a message with no hand, so that its filter starts afresh and the command is the pose of this image, not blended with the last one). The button stays locked until the fingers have reached their targets, plus 2 s, so one press moves the hand once. A second press, or another browser, while a capture is in progress is refused. While it waits, `hand_cam` sends status requests every 100 ms; they move nothing. The waits are bounded: 1 s for `mimic_ctrl` to answer at all, 6 s for the fingers to arrive (then the page says they did not), each followed by the 2 s. A dry run reports no finger positions (and neither does a hand without position feedback), so there is nothing to wait for and the button unlocks 2 s after `mimic_ctrl` confirmed the command.

There are no `O`/`F`/`R` keys in this mode: `mimic_ctrl` keeps the calibration it loaded, which is `hand_calibration.json` if that file exists, else the built-in defaults. To force the defaults, give `mimic_ctrl` a file that does not exist, for example `DH116_IFACE=enp0s31f6 CTRL_ARGS="--calibration none.json" ./scripts/run.sh --web 8080 --snapshot`; the last `--calibration` wins and the file is not created. `--frames`, `--det-every`, `--freeze-crop` and `--profile` have no effect here. The security note above applies even more: anyone who reaches the port can press Capture and so move the hand.

`--camera browser` (needs `--web` and `--snapshot`) takes the picture from the camera of the computer whose browser opens the page, so a laptop's webcam can feed a board that has no camera. The page shows that camera live, mirrored like the other previews; nothing is streamed to the board. One press of **Capture** sends one still frame to `hand_cam` as a JPEG, and everything after that is as described above: the two nets run once on it, the landmarks are drawn on it, one command goes to `mimic_ctrl`, and the button unlocks when the fingers have arrived plus 2 s. `--hand` (the left hand unless you say otherwise) decides which hand counts: a picture of the other one gets "No left hand found (a right hand was seen)" and nothing is sent. `--width` and `--height` are the size asked from the browser's camera (it may pick a nearby one), and `--flip-input` mirrors the picture as for a board camera.

A browser hands its camera only to pages from `https://` or from `http://localhost`, so on `http://<board-ip>:8008/` the page says that the camera is not available. Make the page `localhost` with an ssh tunnel, started on the laptop and kept open:

```bash
ssh -L 8008:localhost:8008 user@<board-ip>
# then open http://localhost:8008/ in the laptop's browser and allow the camera
```

The laptop and the board need an IP link besides the EtherCAT cable, which is not a network: a second Ethernet port or a USB Ethernet adapter on the board, or Wi-Fi. Through a switch, fixed addresses on a private subnet with no gateway are enough, for example `192.168.50.1/24` on the laptop and `192.168.50.2/24` on the board. Do not plug the hand's EtherCAT cable into that switch, and give `DH116_IFACE` the EtherCAT port only.

`hand_cam` takes a JPEG of at most 16 MiB and 16 megapixels that arrives within 5 s, and refuses anything else with a short message that the page shows (for example "The picture must be a JPEG."). The security note above applies once more, in a stronger form: anyone who can reach the port can upload a picture and so choose the pose the hand takes. The ssh tunnel only gives the browser a `localhost` address; the server still listens on every interface, so on a shared network keep the port closed to others with a firewall.

Controller options can be supplied with `CTRL_ARGS`, for example:
```bash
DH116_IFACE=enp0s31f6 CTRL_ARGS="--current 300 --deadband 80" ./scripts/run.sh
```
`mimic_ctrl` options: `--speed` (default 20000), `--current` (per mille; not sent by default, so the hand uses its own limit), `--deadband`, `--min-cutoff` and `--beta` (filter), `--profile`, `--quiet`. The once-per-second log line lists the targets, measured positions, and status bytes in finger order: thumb sweep, thumb bend, index, middle, ring, little. `--profile` adds each joint's current (per mille of the rated current); `./scripts/run.sh --profile` passes it to both programs.

Each joint's status byte (protocol sheet, ECAT): bit 0 enabled, bit 1 position reached, bit 2 zero speed, bit 3 torque reached, bit 4 homing completed, bits 5-7 the state (0 stopped, 1 running, 2 alarm stop, 3 / 4 / 5 at the + / - / both limits, 6 emergency stop, 7 homing). A still hand shows `07`, a moving one `21` to `27`. At start-up `mimic_ctrl` prints the status bytes and how many joints are homed, and refuses to run if a joint is in alarm stop or emergency stop, or not enabled. While running, a joint line is printed only when its problem starts or ends, with the status byte and current: fatal ones in red at once, warnings in yellow once they last 100 ms (colours only on a terminal). Over EtherCAT the hand reports no fault details (such as under-voltage); the vendor tools show those. The hand's own protections (over/under-voltage, overcurrent, overload, stall, over-temperature) all show up here as alarm stop.

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
# no camera on the board: one picture per press from the laptop's webcam (ssh tunnel, see Run)
DH116_IFACE=eth1 ./run.sh --camera browser --web 8008 --snapshot
```

To start the programs by hand instead:

```bash
./bin/mimic_ctrl --dry-run --calibration hand_calibration.json &
./bin/hand_cam --camera 0 --model-dir models --width 640 --height 480 --threads 4

sudo ./bin/mimic_ctrl eth1 --calibration hand_calibration.json      # hardware; instead of --dry-run
```

`--model-dir` is the directory that holds `palm.onnx` and `landmark.onnx` (default `models`); `--palm-model` and `--landmark-model` name the two files directly. Float models and the Qualcomm w8a8 (uint8) models in `models/Qualcomm-w8a8/` both work; the uint8 models' quantisation constants are fixed in `hand_cam.cpp` (they come from that folder's `metadata.json`). `--freeze-crop F` (default 0.03) keeps the hand crop while the hand box moves less than F of its size, so a still hand is cropped the same way every frame and stays steady; `--freeze-crop 0` re-crops every frame. Lower `--width`, `--height`, `--fps`, or `--det-every` rates if the frame rate is too low, and match `--threads` to the board's core count.

To estimate inference time on any device before building anything, run `python3 scripts/bench_models.py` (needs only `pip install onnxruntime numpy`). It times both networks for several thread counts and prints the expected per-frame model cost; `--threads 1,2,4,8` sweeps counts and `taskset -c ...` pins it to chosen cores.

`scripts/quantize_models.py --source <clip or images or camera index>` (dev host; needs `onnx`, `opencv-python-headless` too) writes int8 versions to `models/int8/` and reports accuracy against fp32. Time them on the device with `bench_models.py --model-dir models/int8`. To time the ONNX Runtime that `hand_cam` is actually linked against (e.g. the RVV build), run `./bin/hand_cam --bench --threads 8 [--model-dir models/int8]`; it needs no camera. This is experimental: with sample images only, the palm net still loses the hand in a fifth of frames and the landmark world error is ~5 mm, so calibrate on a real clip before trusting it. 8-bit activations make the landmarks noisy frame to frame, so the robot fingers jitter on a still hand; `--act u16` (w8a16, output heads kept fp32 by default) avoids that: `scripts/quantize_models.py --source ... --act u16 --out-dir models/custom`, then `hand_cam --model-dir models/custom`. `--float-ops`, `--float-nodes`, `--quant-ops`, `--weights`, `--calibrate` and `--list-nodes` choose what is quantized and how; see `quant_comparison.md`. Re-calibrate the hand (open/fist) after switching models.

To compare the accuracy of float and quantized models on a public hand dataset, run `python3 scripts/eval_hand_models.py` (needs `pip install numpy opencv-python onnxruntime`). On first use it downloads the FreiHAND evaluation set (3,960 images with ground-truth joints, about 760 MB) to `datasets/`, then scores the palm detector, the landmark net and the whole palm-to-landmark pipeline for every folder in `models/` that has a palm and a landmark `.onnx` (including the Qualcomm uint8 export). Pass folders explicitly (`models/default models/int8`), use `--limit`/`--stride` for a quick run, and `--json out.json` to save the numbers; `--help` explains each metric.

`--bench` prints the time of each network and a per-frame estimate: `steady` is a frame with the landmark net only, `spike` a frame that also runs the palm net, and `average` spreads the palm net over `--det-every` frames.

`./bin/hand_cam --camera 1 --profile` prints the average time per frame in each stage every 30 frames (through `run.sh`, `--profile` also adds the joint currents to the `mimic_ctrl` log line); use it to see where a slow live run loses its time:

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
| [src/hand_cam.cpp](src/hand_cam.cpp) | Camera capture, palm and landmark networks, preview, and UDP to the controller; `--snapshot` runs one capture and one command per button press, with the picture from the board's camera or, with `--camera browser`, from the browser. |
| [include/web_view.h](include/web_view.h) | `hand_cam --web`: the preview as an MJPEG stream for a web browser, and the calibration keys from it; with `--snapshot`, the Capture page and the state of the capture instead, and with `--camera browser` the upload of the JPEG that the page takes from its own camera. |
| [src/mimic_ctrl.c](src/mimic_ctrl.c) | Controller main loop: UDP from `hand_cam`, landmarks to targets, deadband, log line. |
| [src/dh116.c](src/dh116.c), [include/dh116.h](include/dh116.h) | The hand on EtherCAT (SOEM): start-up, 10 ms cycle, frames sent to the hand, feedback and joint status, opening the fingers, shutdown. |
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