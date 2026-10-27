#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/build.sh [options]

Builds everything the demo needs:
  1. initializes the third_party submodules (SOEM, onnxruntime) when empty
  2. builds ONNX Runtime from third_party/onnxruntime (first run only; slow)
  3. builds dh116_teleop and hand_cam with CMake

Options:
  -j, --jobs N         Parallel jobs (default: min(CPU count, RAM / 3 GB))
  --build-type TYPE    Release (default), Debug, RelWithDebInfo or MinSizeRel
  --ort-root PATH      Use an existing ONNX Runtime prefix (lib/ and include/)
                       instead of building the submodule
  --controller-only    Build only dh116_teleop (no OpenCV or ONNX Runtime)
  --clean              Delete the project build directory (<root>/native) first
  --clean-ort          Delete the ONNX Runtime build and prefix first
  -h, --help           Show this help

Environment:
  DH116_BUILD_ROOT     Root of all build output (default: <repo>/build). The project
                       builds in <root>/native; ONNX Runtime in <root>/deps/native/ort
                       and is installed to <root>/prefix/native
  DH116_BUILD_TYPE     Same as --build-type
  ONNXRUNTIME_ROOT     Same as --ort-root
  PYTHON               Python 3.10+ interpreter used for the ONNX Runtime build
EOF
}

log() { printf '==> %s\n' "$*"; }

die() {
    printf 'build.sh: %s\n' "$*" >&2
    exit 2
}

on_error() {
    printf '\nbuild.sh: failed near line %s. Fix the error above and re-run ./scripts/build.sh; finished steps are skipped.\n' "$1" >&2
}
trap 'on_error $LINENO' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BUILD_ROOT="${DH116_BUILD_ROOT:-$REPO_ROOT/build}"
BUILD_TYPE="${DH116_BUILD_TYPE:-Release}"
ORT_ROOT="${ONNXRUNTIME_ROOT:-}"
PYTHON="${PYTHON:-python3}"
JOBS=""
CONTROLLER_ONLY=0
CLEAN=0
CLEAN_ORT=0

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
        --build-type)
            (($# >= 2)) || die "--build-type requires a value"
            BUILD_TYPE="$2"
            shift 2
            ;;
        --ort-root)
            (($# >= 2)) || die "--ort-root requires a path"
            ORT_ROOT="$2"
            shift 2
            ;;
        --controller-only)
            CONTROLLER_ONLY=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --clean-ort)
            CLEAN_ORT=1
            shift
            ;;
        *)
            die "unknown option: $1 (see --help)"
            ;;
    esac
done

[[ "$BUILD_ROOT" == /* ]] || BUILD_ROOT="$REPO_ROOT/$BUILD_ROOT"
BUILD_DIR="$BUILD_ROOT/native"
ORT_BUILD_DIR="$BUILD_ROOT/deps/native/ort"
ORT_PREFIX="$BUILD_ROOT/prefix/native"

case "$BUILD_TYPE" in
    Release|Debug|RelWithDebInfo|MinSizeRel) ;;
    *) die "invalid build type: $BUILD_TYPE" ;;
esac

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

check_toolchain() {
    local cmake_version
    command -v cmake >/dev/null || die "cmake 3.28 or newer is required"
    cmake_version="$(cmake --version | awk 'NR==1 {print $3}')"
    version_ge "$cmake_version" 3.28 || die "cmake 3.28 or newer is required (found $cmake_version)"
    command -v "${CC:-cc}" >/dev/null || die "a C compiler is required (sudo apt install build-essential)"
    command -v "${CXX:-c++}" >/dev/null || die "a C++ compiler is required (sudo apt install build-essential)"
}

check_python() {
    command -v "$PYTHON" >/dev/null || die "$PYTHON not found; ONNX Runtime needs Python 3.10 or newer"
    "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
        die "ONNX Runtime needs Python 3.10 or newer (set PYTHON=/path/to/python3)"
}

init_submodules() {
    local missing=()
    [[ -f "$SOEM_SRC/CMakeLists.txt" ]] || missing+=(third_party/SOEM)
    if ((! CONTROLLER_ONLY)) && [[ -z "$ORT_ROOT" ]] && [[ ! -f "$ORT_SRC/cmake/CMakeLists.txt" ]]; then
        missing+=(third_party/onnxruntime)
    fi
    ((${#missing[@]})) || return 0

    command -v git >/dev/null || die "git is required to fetch: ${missing[*]}"
    git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
        die "${missing[*]} is empty and this is not a git checkout"
    log "Fetching submodules: ${missing[*]}"
    # ONNX Runtime downloads its own dependencies, so nested submodules are not needed.
    git -C "$REPO_ROOT" submodule update --init -- "${missing[@]}"
}

# Mirrors the lookup in CMakeLists.txt.
ort_prefix_valid() {
    local root="$1" inc
    compgen -G "$root/lib/libonnxruntime.so*" >/dev/null ||
        compgen -G "$root/lib64/libonnxruntime.so*" >/dev/null ||
        return 1
    for inc in include include/onnxruntime include/onnxruntime/core/session include/core/session; do
        [[ -f "$root/$inc/onnxruntime_cxx_api.h" ]] && return 0
    done
    return 1
}

# Fail before the long ONNX Runtime build if the camera frontend cannot link.
check_opencv() {
    local probe
    probe="$(mktemp -d)"
    printf '%s\n' \
        'cmake_minimum_required(VERSION 3.20)' \
        'project(probe LANGUAGES CXX)' \
        'find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)' \
        >"$probe/CMakeLists.txt"
    if ! cmake -S "$probe" -B "$probe/build" >"$probe/log" 2>&1; then
        sed 's/^/    /' "$probe/log" >&2
        rm -rf -- "$probe"
        die "OpenCV was not found (sudo apt install libopencv-dev, or set OpenCV_DIR)"
    fi
    rm -rf -- "$probe"
}

clean_build_dirs() {
    if ((CLEAN)); then
        if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
            log "Removing $BUILD_DIR"
            rm -rf -- "$BUILD_DIR"
        else
            log "No CMake build directory at $BUILD_DIR; nothing to clean"
        fi
    fi
    if ((CLEAN_ORT)); then
        if [[ -d "$ORT_BUILD_DIR" || -d "$ORT_PREFIX" ]]; then
            log "Removing $ORT_BUILD_DIR and $ORT_PREFIX"
            rm -rf -- "$ORT_BUILD_DIR" "$ORT_PREFIX"
        else
            log "No ONNX Runtime build under $BUILD_ROOT; nothing to clean"
        fi
    fi
}

build_ort() {
    local rev built
    rev="$(git -C "$ORT_SRC" rev-parse HEAD 2>/dev/null || cat "$ORT_SRC/VERSION_NUMBER")"

    if ort_prefix_valid "$ORT_PREFIX" && [[ "$(cat "$ORT_PREFIX/.source-rev" 2>/dev/null || true)" == "$rev" ]]; then
        log "ONNX Runtime is already built: $ORT_PREFIX"
        return 0
    fi

    check_python
    log "Building ONNX Runtime from $ORT_SRC with $JOBS jobs (the first build takes a long time)"
    mkdir -p "$ORT_BUILD_DIR"

    "$PYTHON" "$ORT_SRC/tools/ci_build/build.py" \
        --build_dir "$ORT_BUILD_DIR" \
        --config Release \
        --build_shared_lib \
        --parallel "$JOBS" \
        --skip_tests \
        --skip_submodule_sync \
        --compile_no_warning_as_error \
        --update --build \
        --target onnxruntime \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF

    built="$ORT_BUILD_DIR/Release"
    compgen -G "$built/libonnxruntime.so*" >/dev/null ||
        die "ONNX Runtime build finished without producing $built/libonnxruntime.so"

    log "Installing ONNX Runtime into $ORT_PREFIX"
    rm -rf -- "$ORT_PREFIX"
    mkdir -p "$ORT_PREFIX/lib" "$ORT_PREFIX/include/onnxruntime"
    cp -a "$built"/libonnxruntime.so* "$ORT_PREFIX/lib/"
    [[ ! -f "$built/libonnxruntime_providers_shared.so" ]] ||
        cp -a "$built/libonnxruntime_providers_shared.so" "$ORT_PREFIX/lib/"
    cp -a "$ORT_SRC"/include/onnxruntime/core/session/onnxruntime_*.h \
        "$ORT_SRC"/include/onnxruntime/core/session/onnxruntime_*.inc \
        "$ORT_PREFIX/include/onnxruntime/"
    printf '%s\n' "$rev" >"$ORT_PREFIX/.source-rev"
}

build_project() {
    local targets=(dh116_teleop) cmake_args=(
        -S "$REPO_ROOT"
        -B "$BUILD_DIR"
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    )

    if ((CONTROLLER_ONLY)); then
        cmake_args+=(-DBUILD_HAND_CAM=OFF)
    else
        targets+=(hand_cam)
        # -U drops cached lookups so a changed ONNX Runtime prefix takes effect.
        cmake_args+=(
            -DBUILD_HAND_CAM=ON
            -DONNXRUNTIME_ROOT="$ORT_ROOT"
            -UONNXRUNTIME_LIBRARY
            -UONNXRUNTIME_INCLUDE_DIR
        )
    fi

    log "Configuring project ($BUILD_TYPE) in $BUILD_DIR"
    cmake "${cmake_args[@]}"
    log "Building: ${targets[*]}"
    cmake --build "$BUILD_DIR" --target "${targets[@]}" --parallel "$JOBS"
}

check_toolchain
clean_build_dirs
init_submodules

if ((! CONTROLLER_ONLY)); then
    check_opencv
    if [[ -n "$ORT_ROOT" ]]; then
        [[ "$ORT_ROOT" == /* ]] || ORT_ROOT="$(cd -- "$ORT_ROOT" 2>/dev/null && pwd)" ||
            die "ONNX Runtime prefix does not exist: $ORT_ROOT"
        ort_prefix_valid "$ORT_ROOT" ||
            die "no libonnxruntime.so and onnxruntime_cxx_api.h under $ORT_ROOT (expected lib/ and include/)"
        log "Using existing ONNX Runtime: $ORT_ROOT"
    else
        build_ort
        ORT_ROOT="$ORT_PREFIX"
    fi
fi

build_project

log "Build complete"
printf '  controller: %s/dh116_teleop\n' "$BUILD_DIR"
if ((! CONTROLLER_ONLY)); then
    printf '  camera:     %s/hand_cam\n' "$BUILD_DIR"
fi
printf '  next:       DH116_IFACE=<ethercat-interface> ./scripts/run.sh   (or ./scripts/run.sh --dry-run)\n'
