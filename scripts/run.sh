#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/run.sh [--dry-run] [--camera N|browser] [--port N] [hand_cam options]

Starts mimic_ctrl (EtherCAT controller) and hand_cam (camera + neural nets) together;
when either one exits, the other is stopped.

Examples:
  ./scripts/run.sh --dry-run                       camera and inference only, no EtherCAT, no sudo
  DH116_IFACE=enp0s31f6 ./scripts/run.sh           drive the hand (see `ip -brief link`)
  ./scripts/run.sh --dry-run --fps 15 --web 8080   any other option goes to hand_cam (see hand_cam --help)
  ./scripts/run.sh --dry-run --web 8080 --snapshot not real time: one image per Capture press
  ./scripts/run.sh --dry-run --camera browser --web 8008 --snapshot
                                                   the picture comes from the webcam of the computer whose
                                                   browser opens the page (ssh -L 8008:localhost:8008 user@board)
  ./scripts/run.sh --dry-run --palm-model a.onnx --landmark-model b.onnx

Options (everything else is passed to hand_cam):
  --dry-run            no EtherCAT
  --camera N           camera index (default 0), same as DH116_CAMERA; "browser" needs --web and --snapshot
  --port N             UDP port shared by both programs (default 5006), same as MIMIC_PORT
  --profile            goes to both: hand_cam prints its stage times, mimic_ctrl logs the joint currents

run.sh starts hand_cam with --width 640 --height 480 --threads 4 --det-every 30 --model-dir <repo>/models.
Repeat any of these in your options to change it: the last value wins.

Environment:
  DH116_IFACE          EtherCAT interface; required unless --dry-run
  CTRL_ARGS            Extra mimic_ctrl arguments, e.g. "--current 300 --deadband 80"
  DH116_BUILD_ROOT     Build root shared with build.sh (default: <repo>/build)

If an executable is missing, scripts/build.sh runs first; after a source change, run it yourself.
Inside a riscv64 bundle the executables come from bin/ and nothing is built.
EOF
}

die() {
    printf 'run.sh: %s\n' "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE=0 # 1 inside the riscv64 tarball, where run.sh sits next to bin/ instead of in scripts/
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
PORT="${MIMIC_PORT:-5006}"
CAMERA="${DH116_CAMERA:-0}"
DRY_RUN=0
PROFILE=0
CAMERA_ARGS=()
CTRL_EXTRA=()

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
        --profile)
            PROFILE=1
            CAMERA_ARGS+=("$1")
            shift
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
    die "MIMIC_PORT/--port must be between 1 and 65535"
fi

SUDO=()
if (( ! DRY_RUN && EUID != 0 )); then
    command -v sudo >/dev/null || die "sudo is required for EtherCAT access (or run as root)"
    SUDO=(sudo -n)
fi

CONTROLLER="$BUILD_DIR/mimic_ctrl"
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

if [[ -n "${CTRL_ARGS:-}" ]]; then
    read -r -a CTRL_EXTRA <<< "$CTRL_ARGS"
fi
if (( PROFILE )); then
    CTRL_EXTRA+=(--profile)
fi

if (( DRY_RUN )); then
    CTRL_CMD=("$CONTROLLER" --dry-run --port "$PORT" --calibration "$ROOT_DIR/hand_calibration.json" "${CTRL_EXTRA[@]}")
else
    CTRL_CMD=("${SUDO[@]}" "$CONTROLLER" "$IFACE" --port "$PORT" --calibration "$ROOT_DIR/hand_calibration.json" "${CTRL_EXTRA[@]}")
fi

# Defaults first, user options last: the last value of a repeated option wins.
CAMERA_CMD=(
    "$CAMERA_APP"
    --camera "$CAMERA"
    --port "$PORT"
    --model-dir "$ROOT_DIR/models"
    --width 640
    --height 480
    --threads 4
    --det-every 30
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