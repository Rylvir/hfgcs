#!/bin/sh
# Deploy the HFGCS checkout and restart only what changed (see docs/deploy.md).
# Use this instead of `systemctl --user restart hfgcs-...`: OpenWebRX stops the SDR when
# its only client (hfgcs-8992) disconnects, so every needless restart is an audio gap.
#   deploy/deploy.sh --dry-run    # show the plan
#   deploy/deploy.sh              # fast-forward to origin/main, restart affected units
#   deploy/deploy.sh --help
set -eu
if [ "$(id -u)" = 0 ]; then
    echo "deploy.sh: do not run as root or with sudo; run it as the service user" >&2
    exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
exec "$here/bin/hfgcs-python" "$repo/scripts/hfgcs_deploy.py" "$@"
