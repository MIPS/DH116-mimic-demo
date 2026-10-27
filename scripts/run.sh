#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/run.sh [--dry-run] [camera options]

Runs the C++ camera frontend and the C EtherCAT controller. Camera options are
passed to hand_cam, for example: --hand right --camera 0 --flip-input.

If either executable is missing in the repository checkout, scripts/build.sh runs
first. To rebuild after a source change, run scripts/build.sh yourself.

Inside a riscv64 bundle the executables come from bin/ and nothing is built.

Environment:
  DH116_IFACE          EtherCAT interface; required unless --dry-run (e.g. enp0s31f6)
  DH116_CAMERA         Default camera source (default: 0)
  TELEOP_PORT          UDP port shared by both processes (default: 5006)
  TELEOP_ARGS          Extra controller arguments, e.g. "--cur 600 --deadband 80"
  DH116_BUILD_ROOT     Build root shared with build.sh (default: <repo>/build)

Build options such as DH116_BUILD_TYPE and ONNXRUNTIME_ROOT are read by
scripts/build.sh; see scripts/build.sh --help.
EOF
}

die() {
    printf 'run.sh: %s\n' "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE=0
if [[ -f "$SCRIPT_DIR/../CMakeLists.txt" ]]; then
    ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
else
    ROOT_DIR="$SCRIPT_DIR"
    BUNDLE=1
fi
BUILD_ROOT="${DH116_BUILD_ROOT:-$ROOT_DIR/build}"
[[ "$BUILD_ROOT" == /* ]] || BUILD_ROOT="$ROOT_DIR/$BUILD_ROOT"
BUILD_DIR="$BUILD_ROOT/native"
if (( BUNDLE )); then
    BUILD_DIR="$ROOT_DIR/bin"
fi
IFACE="${DH116_IFACE:-}"
PORT="${TELEOP_PORT:-5006}"
CAMERA="${DH116_CAMERA:-0}"
DRY_RUN=0
CAMERA_ARGS=()
CTRL_ARGS=()

while (($#)); do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --camera)
            (($# >= 2)) || die "--camera requires a source"
            CAMERA="$2"
            shift 2
            ;;
        --port)
            (($# >= 2)) || die "--port requires a number"
            PORT="$2"
            shift 2
            ;;
        --)
            shift
            CAMERA_ARGS+=("$@")
            break
            ;;
        *)
            CAMERA_ARGS+=("$1")
            shift
            ;;
    esac
done

if (( ! DRY_RUN )) && [[ -z "$IFACE" ]]; then
    die "set DH116_IFACE to the wired EtherCAT interface, or use --dry-run"
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
    die "TELEOP_PORT/--port must be between 1 and 65535"
fi

SUDO=()
if (( ! DRY_RUN && EUID != 0 )); then
    command -v sudo >/dev/null || die "sudo is required for EtherCAT access (or run as root)"
    SUDO=(sudo -n)
fi

CONTROLLER="$BUILD_DIR/dh116_teleop"
CAMERA_APP="$BUILD_DIR/hand_cam"

if (( BUNDLE )); then
    [[ -x "$CONTROLLER" && -x "$CAMERA_APP" ]] || die "missing executables in $BUILD_DIR"
elif [[ ! -x "$CONTROLLER" || ! -x "$CAMERA_APP" ]]; then
    echo "run.sh: executables not found in $BUILD_DIR; running build.sh" >&2
    DH116_BUILD_ROOT="$BUILD_ROOT" "$ROOT_DIR/scripts/build.sh"
fi
[[ -x "$CONTROLLER" ]] || die "controller executable was not produced: $CONTROLLER"
[[ -x "$CAMERA_APP" ]] || die "camera executable was not produced: $CAMERA_APP"

if ((${#SUDO[@]})); then
    sudo -v
fi

# Keep V4L2 cameras at their requested frame rate when the driver supports it.
DYN_FPS=""
CAM_DEV=""
if [[ "$CAMERA" =~ ^[0-9]+$ ]] && command -v v4l2-ctl >/dev/null; then
    CAM_DEV="/dev/video$CAMERA"
    DYN_FPS="$(v4l2-ctl -d "$CAM_DEV" --get-ctrl exposure_dynamic_framerate 2>/dev/null | awk '{print $2}')" || true
    if [[ -n "$DYN_FPS" ]]; then
        v4l2-ctl -d "$CAM_DEV" --set-ctrl exposure_dynamic_framerate=0 2>/dev/null || DYN_FPS=""
    fi
fi

if [[ -n "${TELEOP_ARGS:-}" ]]; then
    read -r -a CTRL_ARGS <<< "$TELEOP_ARGS"
fi

if (( DRY_RUN )); then
    CTRL_CMD=("$CONTROLLER" --dry-run --port "$PORT" --calibration "$ROOT_DIR/hand_calibration.json" "${CTRL_ARGS[@]}")
else
    CTRL_CMD=("${SUDO[@]}" "$CONTROLLER" "$IFACE" --port "$PORT" --calibration "$ROOT_DIR/hand_calibration.json" "${CTRL_ARGS[@]}")
fi

CAMERA_CMD=(
    "$CAMERA_APP"
    --camera "$CAMERA"
    --port "$PORT"
    --model-dir "$ROOT_DIR/models"
    --width 640
    --height 480
    --threads 4
    --det-every 15
    "${CAMERA_ARGS[@]}"
)

CTRL_PID=""
CAMERA_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM

    for pid in "$CAMERA_PID" "$CTRL_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
        fi
    done

    for pid in "$CAMERA_PID" "$CTRL_PID"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done

    if [[ -n "$DYN_FPS" && -n "$CAM_DEV" ]]; then
        v4l2-ctl -d "$CAM_DEV" --set-ctrl "exposure_dynamic_framerate=$DYN_FPS" 2>/dev/null || true
    fi

    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${CTRL_CMD[@]}" &
CTRL_PID=$!
"${CAMERA_CMD[@]}" &
CAMERA_PID=$!

# Either process ending shuts down the other through cleanup. The camera sends
# quit packets on normal exit; SIGINT gives the controller the same safe path.
wait -n "$CTRL_PID" "$CAMERA_PID"