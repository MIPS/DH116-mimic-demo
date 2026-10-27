#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/build-riscv.sh [options]

Cross-compiles the demo into a static riscv64 (RVA23, RVV 1.0) bundle:
  1. finds or downloads a riscv64-linux-gnu GCC toolchain (no root needed)
  2. builds static OpenCV (headless: V4L2 capture, JPEG/PNG)
  3. builds static ONNX Runtime with RISC-V Vector kernels
  4. builds dh116_teleop and hand_cam
  5. packs dist/dh116-riscv64-rva23.tar.gz

Finished steps are skipped on later runs.

Options:
  -j, --jobs N         Parallel jobs (default: min(CPU count, RAM / 3 GB))
  --controller-only    Build only dh116_teleop (no OpenCV or ONNX Runtime)
  --test               After building, smoke-test the bundle under qemu-user
                       (-cpu rva23u64,vlen=1024)
  --clean              Delete the project build and the bundle first
  --clean-deps         Also delete the OpenCV and ONNX Runtime builds
  -h, --help           Show this help

Environment:
  DH116_BUILD_ROOT     Root of all build output (default: <repo>/build); the bundle
                       is written to <repo>/dist
  PYTHON               Python 3.10+ interpreter used for the ONNX Runtime build
  QEMU                 qemu-user binary for --test (default: qemu-riscv64)
EOF
}

log() { printf '==> %s\n' "$*"; }

die() {
    printf 'build-riscv.sh: %s\n' "$*" >&2
    exit 2
}

on_error() {
    printf '\nbuild-riscv.sh: failed near line %s. Fix the error above and re-run; finished steps are skipped.\n' "$1" >&2
}
trap 'on_error $LINENO' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BUILD_ROOT="${DH116_BUILD_ROOT:-$REPO_ROOT/build}"
[[ "$BUILD_ROOT" == /* ]] || BUILD_ROOT="$REPO_ROOT/$BUILD_ROOT"
PYTHON="${PYTHON:-python3}"
QEMU="${QEMU:-qemu-riscv64}"
JOBS=""
CONTROLLER_ONLY=0
RUN_TESTS=0
CLEAN=0
CLEAN_DEPS=0

TOOLCHAIN_FILE="$REPO_ROOT/cmake/toolchains/riscv64-rva23-static.cmake"
OPENCV_VERSION="4.10.0"
OPENCV_SHA256="b2171af5be6b26f7a06b1229948bbb2bdaa74fcf5cd097e0af6378fce50a6eb9"
OPENCV_URL="https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VERSION.tar.gz"

TOOLCHAIN_DIR="$BUILD_ROOT/toolchain/riscv64"
PREFIX="$BUILD_ROOT/prefix/riscv64"
LOGS="$BUILD_ROOT/logs/riscv64"
OPENCV_SRC="$BUILD_ROOT/src/opencv-$OPENCV_VERSION"
OPENCV_BUILD="$BUILD_ROOT/deps/riscv64/opencv"
ORT_BUILD="$BUILD_ROOT/deps/riscv64/ort"
PROJECT_BUILD="$BUILD_ROOT/riscv64"
BUNDLE_NAME="dh116-riscv64-rva23"
DIST="$REPO_ROOT/dist"
BUNDLE="$DIST/$BUNDLE_NAME"
SOEM_SRC="$REPO_ROOT/third_party/SOEM"
ORT_SRC="$REPO_ROOT/third_party/onnxruntime"

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -j|--jobs)
            (($# >= 2)) || die "$1 requires a number"
            JOBS="$2"
            shift 2
            ;;
        --controller-only)
            CONTROLLER_ONLY=1
            shift
            ;;
        --test)
            RUN_TESTS=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --clean-deps)
            CLEAN=1
            CLEAN_DEPS=1
            shift
            ;;
        *)
            die "unknown option: $1 (see --help)"
            ;;
    esac
done

default_jobs() {
    local cpus mem_kb by_mem
    cpus="$(nproc 2>/dev/null || echo 1)"
    mem_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    by_mem=$((mem_kb / 1024 / 1024 / 3))
    ((by_mem >= 1)) || by_mem=1
    echo $((cpus < by_mem ? cpus : by_mem))
}

if [[ -z "$JOBS" ]]; then
    JOBS="$(default_jobs)"
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "jobs must be a positive integer"

# True when $1 >= $2 under version ordering.
version_ge() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

# Runs a long command with its output in $LOGS/<name>.log; prints the tail on failure.
run_logged() {
    local name="$1"
    shift
    mkdir -p "$LOGS"
    log "$name (log: $LOGS/$name.log)"
    if ! "$@" >"$LOGS/$name.log" 2>&1; then
        tail -n 40 "$LOGS/$name.log" >&2
        die "$name failed; full log: $LOGS/$name.log"
    fi
}

check_host() {
    local cmake_version tool
    command -v cmake >/dev/null || die "cmake 3.28 or newer is required"
    cmake_version="$(cmake --version | awk 'NR==1 {print $3}')"
    version_ge "$cmake_version" 3.28 || die "cmake 3.28 or newer is required (found $cmake_version)"
    for tool in curl tar sha256sum git; do
        command -v "$tool" >/dev/null || die "$tool is required"
    done
    GENERATOR=()
    ORT_GENERATOR=()
    if command -v ninja >/dev/null; then
        GENERATOR=(-G Ninja)
        ORT_GENERATOR=(--cmake_generator Ninja)
    fi
}

# Debian/Ubuntu only: unpack the cross compiler .debs into $TOOLCHAIN_DIR.
bootstrap_toolchain() {
    local pkgs download deb
    command -v apt-get >/dev/null && command -v dpkg-deb >/dev/null ||
        die "no riscv64-linux-gnu-gcc found; install a riscv64 cross GCC 14 or newer and put it on PATH"

    pkgs="$(apt-get install -s --no-install-recommends gcc-riscv64-linux-gnu g++-riscv64-linux-gnu 2>/dev/null |
        awk '/^Inst / && ($2 ~ /riscv64|cross-base$/) {print $2}')"
    [[ -n "$pkgs" ]] || die "apt cannot resolve gcc-riscv64-linux-gnu (try: sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu)"

    log "Downloading the riscv64 cross toolchain into $TOOLCHAIN_DIR (no root needed)"
    download="$(mktemp -d)"
    # shellcheck disable=SC2086 # package names never contain spaces
    (cd "$download" && apt-get download $pkgs >/dev/null)
    mkdir -p "$TOOLCHAIN_DIR"
    for deb in "$download"/*.deb; do
        dpkg-deb -x "$deb" "$TOOLCHAIN_DIR"
    done
    rm -rf -- "$download"
}

use_local_toolchain() {
    local multiarch
    multiarch="$("${CC:-cc}" -dumpmachine)"
    export PATH="$TOOLCHAIN_DIR/usr/bin:$PATH"
    # The unpacked binutils load their private libraries from here.
    export LD_LIBRARY_PATH="$TOOLCHAIN_DIR/usr/lib/$multiarch${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

setup_toolchain() {
    local major
    if ! command -v riscv64-linux-gnu-gcc >/dev/null; then
        [[ -x "$TOOLCHAIN_DIR/usr/bin/riscv64-linux-gnu-gcc" ]] || bootstrap_toolchain
        use_local_toolchain
    fi
    command -v riscv64-linux-gnu-g++ >/dev/null || die "riscv64-linux-gnu-g++ not found"
    major="$(riscv64-linux-gnu-gcc -dumpversion | cut -d. -f1)"
    ((major >= 14)) || die "GCC 14 or newer is required for the RVV 1.0 intrinsics (found $major)"
    log "Cross compiler: $(command -v riscv64-linux-gnu-gcc) (GCC $major)"
}

clean_dirs() {
    ((CLEAN)) || return 0
    log "Removing the project build and bundle"
    rm -rf -- "$PROJECT_BUILD" "$BUNDLE" "$BUNDLE.tar.gz" "$BUNDLE.tar.gz.sha256"
    if ((CLEAN_DEPS)); then
        log "Removing the OpenCV and ONNX Runtime builds"
        rm -rf -- "$OPENCV_BUILD" "$ORT_BUILD" "$PREFIX"
    fi
}

init_submodules() {
    local missing=()
    [[ -f "$SOEM_SRC/CMakeLists.txt" ]] || missing+=(third_party/SOEM)
    if ((! CONTROLLER_ONLY)) && [[ ! -f "$ORT_SRC/cmake/CMakeLists.txt" ]]; then
        missing+=(third_party/onnxruntime)
    fi
    ((${#missing[@]})) || return 0

    git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
        die "${missing[*]} is empty and this is not a git checkout"
    log "Fetching submodules: ${missing[*]}"
    git -C "$REPO_ROOT" submodule update --init -- "${missing[@]}"
}

build_opencv() {
    local tarball="$BUILD_ROOT/src/opencv-$OPENCV_VERSION.tar.gz"

    if [[ -f "$PREFIX/lib/cmake/opencv4/OpenCVConfig.cmake" && -f "$PREFIX/lib/libopencv_videoio.a" ]]; then
        log "OpenCV is already built: $PREFIX"
        return 0
    fi

    if [[ ! -f "$OPENCV_SRC/CMakeLists.txt" ]]; then
        log "Downloading OpenCV $OPENCV_VERSION"
        mkdir -p "$OPENCV_SRC"
        curl -fL --retry 3 -o "$tarball" "$OPENCV_URL"
        printf '%s  %s\n' "$OPENCV_SHA256" "$tarball" | sha256sum -c --quiet - ||
            die "checksum mismatch for $tarball"
        tar -xzf "$tarball" -C "$OPENCV_SRC" --strip-components=1
        rm -f -- "$tarball"
    fi

    # CV_RVV_SCALABLE selects OpenCV's RVV 1.0 vector-length-agnostic intrinsics.
    run_logged opencv-configure cmake -S "$OPENCV_SRC" -B "$OPENCV_BUILD" "${GENERATOR[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_C_FLAGS=-DCV_RVV_SCALABLE \
        -DCMAKE_CXX_FLAGS=-DCV_RVV_SCALABLE \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_LIST=core,imgproc,videoio \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
        -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF \
        -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DWITH_JPEG=ON \
        -DWITH_V4L=ON -DWITH_ADE=OFF \
        -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF \
        -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
        -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_PROTOBUF=OFF \
        -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_OPENMP=OFF \
        -DENABLE_PRECOMPILED_HEADERS=OFF
    run_logged opencv-build cmake --build "$OPENCV_BUILD" --parallel "$JOBS"
    run_logged opencv-install cmake --install "$OPENCV_BUILD"
}

# Merges the static libraries of the ONNX Runtime build tree into one archive.
merge_ort_archives() {
    local out="$PREFIX/lib/libonnxruntime.a" mri="$ORT_BUILD/merge.mri" archive
    mkdir -p "$PREFIX/lib"
    rm -f -- "$out"
    {
        printf 'CREATE %s\n' "$out"
        while IFS= read -r archive; do
            printf 'ADDLIB %s\n' "$archive"
        done < <(find "$ORT_BUILD/Release" -name '*.a' | sort)
        printf 'SAVE\nEND\n'
    } >"$mri"
    riscv64-linux-gnu-ar -M <"$mri"
    riscv64-linux-gnu-ranlib "$out"
}

build_ort() {
    local rev
    rev="$(git -C "$ORT_SRC" rev-parse HEAD 2>/dev/null || cat "$ORT_SRC/VERSION_NUMBER")"

    if [[ -f "$PREFIX/lib/libonnxruntime.a" && "$(cat "$PREFIX/.ort-rev" 2>/dev/null || true)" == "$rev" ]]; then
        log "ONNX Runtime is already built: $PREFIX/lib/libonnxruntime.a"
        return 0
    fi

    command -v "$PYTHON" >/dev/null || die "$PYTHON not found; ONNX Runtime needs Python 3.10 or newer"
    "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
        die "ONNX Runtime needs Python 3.10 or newer (set PYTHON=/path/to/python3)"

    # --rv64 supplies a toolchain file that expects a vendor toolchain layout;
    # the later CMAKE_TOOLCHAIN_FILE define replaces it with ours.
    run_logged ort-build "$PYTHON" "$ORT_SRC/tools/ci_build/build.py" \
        --build_dir "$ORT_BUILD" \
        --config Release \
        "${ORT_GENERATOR[@]}" \
        --parallel "$JOBS" \
        --skip_tests \
        --skip_submodule_sync \
        --compile_no_warning_as_error \
        --rv64 --riscv_toolchain_root "$TOOLCHAIN_DIR/usr" \
        --enable_rvv \
        --update \
        --cmake_extra_defines \
        CMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
        onnxruntime_BUILD_UNIT_TESTS=OFF \
        FLATBUFFERS_BUILD_FLATC=OFF

    # Only the static libraries are needed; the default target would also link a shared provider library.
    run_logged ort-build-libs cmake --build "$ORT_BUILD/Release" --parallel "$JOBS" --target \
        onnxruntime_session onnxruntime_providers onnxruntime_framework onnxruntime_optimizer \
        onnxruntime_graph onnxruntime_util onnxruntime_mlas onnxruntime_common \
        onnxruntime_flatbuffers onnxruntime_lora model_package onnx onnx_proto flatbuffers libprotobuf-lite re2

    log "Installing ONNX Runtime into $PREFIX"
    merge_ort_archives
    mkdir -p "$PREFIX/include/onnxruntime"
    cp -a "$ORT_SRC"/include/onnxruntime/core/session/onnxruntime_*.h \
        "$ORT_SRC"/include/onnxruntime/core/session/onnxruntime_*.inc \
        "$PREFIX/include/onnxruntime/"
    printf '%s\n' "$rev" >"$PREFIX/.ort-rev"
}

build_project() {
    local targets=(dh116_teleop) cmake_args=(
        -S "$REPO_ROOT"
        -B "$PROJECT_BUILD"
        "${GENERATOR[@]}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE"
        -DCMAKE_FIND_ROOT_PATH="$PREFIX"
    )

    if ((CONTROLLER_ONLY)); then
        cmake_args+=(-DBUILD_HAND_CAM=OFF)
    else
        targets+=(hand_cam)
        cmake_args+=(
            -DBUILD_HAND_CAM=ON
            -DHAND_CAM_HEADLESS=ON
            -DONNXRUNTIME_ROOT="$PREFIX"
            -DOpenCV_DIR="$PREFIX/lib/cmake/opencv4"
        )
    fi

    run_logged project-configure cmake "${cmake_args[@]}"
    run_logged project-build cmake --build "$PROJECT_BUILD" --target "${targets[@]}" --parallel "$JOBS"
}

# Static means no PT_INTERP segment, so the binary runs on any riscv64 Linux.
verify_static_riscv64() {
    local exe="$1" header
    header="$(riscv64-linux-gnu-readelf -h "$exe")"
    grep -q 'Machine:.*RISC-V' <<<"$header" || die "$exe is not a RISC-V executable"
    if riscv64-linux-gnu-readelf -l "$exe" | grep -q 'INTERP'; then
        die "$exe is dynamically linked"
    fi
}

make_bundle() {
    local exe
    log "Packing $BUNDLE"
    rm -rf -- "$BUNDLE" "$BUNDLE.tar.gz"
    mkdir -p "$BUNDLE/bin" "$BUNDLE/models"

    for exe in dh116_teleop hand_cam; do
        [[ -f "$PROJECT_BUILD/$exe" ]] || continue
        verify_static_riscv64 "$PROJECT_BUILD/$exe"
        riscv64-linux-gnu-strip --strip-unneeded -o "$BUNDLE/bin/$exe" "$PROJECT_BUILD/$exe"
    done

    cp "$REPO_ROOT/hand_calibration.json" "$SCRIPT_DIR/run.sh" "$BUNDLE/"
    if ((! CONTROLLER_ONLY)); then
        cp "$REPO_ROOT"/models/hand_palm.onnx "$REPO_ROOT"/models/hand_landmark_detector.onnx "$BUNDLE/models/"
    fi

    tar -C "$DIST" -czf "$BUNDLE.tar.gz" "$BUNDLE_NAME"
    (cd "$DIST" && sha256sum "$BUNDLE_NAME.tar.gz" >"$BUNDLE_NAME.tar.gz.sha256")
}

# Writes a short MJPEG clip that OpenCV can read without FFmpeg.
make_test_clip() {
    local clip="$1"
    "$PYTHON" - "$clip" <<'EOF' 2>/dev/null
import sys
import cv2
import numpy as np

writer = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
if not writer.isOpened():
    sys.exit(1)
rng = np.random.default_rng(0)
for _ in range(10):
    writer.write(rng.integers(90, 160, (240, 320, 3), dtype=np.uint8))
writer.release()
EOF
}

smoke_test() {
    local port=5106 cpu="rva23u64,vlen=1024" clip="$LOGS/test-clip.avi" out status=0
    command -v "$QEMU" >/dev/null || die "$QEMU not found (install qemu-user, or set QEMU=/path/to/qemu-riscv64)"
    mkdir -p "$LOGS"

    log "Test: dh116_teleop --dry-run under $QEMU -cpu $cpu"
    out="$LOGS/test-controller.log"
    timeout -s KILL 5 "$QEMU" -cpu "$cpu" "$BUNDLE/bin/dh116_teleop" --dry-run --port "$port" \
        --calibration "$BUNDLE/hand_calibration.json" >"$out" 2>&1 || status=$?
    # timeout stops a healthy controller with 124 or 137; anything else is a failure.
    ((status == 124 || status == 137)) || { cat "$out" >&2; die "controller exited early (status $status)"; }
    grep -q . "$out" || die "controller printed nothing"
    sed 's/^/    /' "$out" | head -n 8

    ((CONTROLLER_ONLY)) && return 0

    log "Test: hand_cam --help"
    "$QEMU" -cpu "$cpu" "$BUNDLE/bin/hand_cam" --help >/dev/null

    if ! make_test_clip "$clip"; then
        log "Skipping the video test: needs $PYTHON with the cv2 and numpy modules"
        return 0
    fi

    log "Test: hand_cam on a synthetic clip against a dry-run controller (slow under emulation)"
    "$QEMU" -cpu "$cpu" "$BUNDLE/bin/dh116_teleop" --dry-run --port "$port" \
        --calibration "$BUNDLE/hand_calibration.json" >"$LOGS/test-controller2.log" 2>&1 &
    local controller=$!
    status=0
    timeout -s KILL 900 "$QEMU" -cpu "$cpu" "$BUNDLE/bin/hand_cam" --camera "$clip" --no-preview \
        --frames 3 --threads 2 --port "$port" --model-dir "$BUNDLE/models" \
        >"$LOGS/test-camera.log" 2>&1 || status=$?
    kill "$controller" 2>/dev/null || true
    wait "$controller" 2>/dev/null || true
    sed 's/^/    /' "$LOGS/test-camera.log" | tail -n 8
    ((status == 0)) || die "hand_cam failed under emulation (status $status); log: $LOGS/test-camera.log"
}

check_host
clean_dirs
setup_toolchain
init_submodules

if ((! CONTROLLER_ONLY)); then
    build_opencv
    build_ort
fi
build_project
make_bundle

if ((RUN_TESTS)); then
    smoke_test
fi

log "Build complete"
printf '  bundle:  %s.tar.gz\n' "$BUNDLE"
printf '  next:    scp it to the board, then: tar xzf %s.tar.gz && cd %s && ./run.sh --dry-run\n' \
    "$BUNDLE_NAME" "$BUNDLE_NAME"
