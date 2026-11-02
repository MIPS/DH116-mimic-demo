#!/usr/bin/env bash
# Home the DH116: index, middle, ring and little together, then the two thumb motors together.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/home.sh [--group auto|fingers|thumb|all] [--method M] [--dir D] [--timeout ms] [--dry-run]

  --group    auto (default): fingers (index, middle, ring, little) together, then the thumb motors together
             fingers / thumb: only that group; all: all six joints at once
  --method   0 torque (default), 1 encoder Z, 2 encoder Z + torque
  --dir      0 positive (closing), 1 negative (opening, default), 2 / 3 the same + stroke recognition
  --timeout  per group, give up after this many ms (default 10000)
  --dry-run  print the command instead of running it

Environment:
  DH116_IFACE   EtherCAT interface; required (e.g. enp0s31f6)

Builds test/home_test first when it is missing (needs scripts/build.sh to have run once). Raw EtherCAT access
needs root, so the tool runs under sudo unless this script already runs as root.
Exit code: 0 homed, 1 usage, 2 start-up failed, 3 homing failed (timeout, Ctrl+C, joint or bus fault).
EOF
}

die() {
    printf 'home.sh: %s\n' "$*" >&2
    exit 1
}

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${DH116_BUILD_ROOT:-$ROOT_DIR/build}"
[[ "$BUILD_ROOT" == /* ]] || BUILD_ROOT="$ROOT_DIR/$BUILD_ROOT"
HOME_TEST="$BUILD_ROOT/native/home_test"

GROUP=auto
METHOD=0
DIR=1
TIMEOUT=10000
DRY_RUN=0

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --group|--method|--dir|--timeout)
            (($# >= 2)) || die "$1 needs a value"
            case "$1" in
                --group) GROUP="$2" ;;
                --method) METHOD="$2" ;;
                --dir) DIR="$2" ;;
                --timeout) TIMEOUT="$2" ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 1 ;;
    esac
done

case "$GROUP" in
    auto|fingers|thumb) GROUP_ARGS=(--group "$GROUP") ;;
    all) GROUP_ARGS=(--joint 0) ;;
    *) die "unknown group '$GROUP' (auto, fingers, thumb, all)" ;;
esac

IFACE="${DH116_IFACE:-}"
[[ -n "$IFACE" ]] || die "set DH116_IFACE to the wired EtherCAT interface"

CMD=("$HOME_TEST" "$IFACE" "${GROUP_ARGS[@]}" --method "$METHOD" --dir "$DIR" --timeout "$TIMEOUT")
if ((EUID != 0)); then
    CMD=(sudo "${CMD[@]}")
fi

if ((DRY_RUN)); then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

if [[ ! -x "$HOME_TEST" ]]; then
    "$ROOT_DIR/test/build.sh"
fi

exec "${CMD[@]}"
