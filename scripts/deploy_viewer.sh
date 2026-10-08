#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    printf 'Run on the target server: sudo bash scripts/deploy_viewer.sh\n' >&2
    exit 1
fi

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SERVICE_USER=${SERVICE_USER:-${SUDO_USER:-root}}
VIEWER_HOST=${VIEWER_HOST:-127.0.0.1}
VIEWER_PORT=${VIEWER_PORT:-8765}
PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_DIR/.venv/bin/python"
    else
        PYTHON_BIN=$(command -v python3)
    fi
fi

if [[ ! -f "$REPO_DIR/kilt_synmes/viewer.py" || ! -f "$REPO_DIR/kilt_synmes/viewer.html" || ! -d "$REPO_DIR/predictions/synmes" ]]; then
    printf 'Copy the viewer files and predictions/synmes directory to this server first.\n' >&2
    exit 1
fi
if [[ ! "$VIEWER_PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$VIEWER_PORT < 1 || 10#$VIEWER_PORT > 65535 )); then
    printf 'VIEWER_PORT must be between 1 and 65535.\n' >&2
    exit 1
fi
if [[ ! "$VIEWER_HOST" =~ ^[a-zA-Z0-9.:_-]+$ || ! "$SERVICE_USER" =~ ^[a-zA-Z0-9_-]+$ || "$PYTHON_BIN" != /* || ! -x "$PYTHON_BIN" ]]; then
    printf 'Invalid host, user, or absolute Python interpreter path.\n' >&2
    exit 1
fi
id "$SERVICE_USER" >/dev/null
command -v systemctl >/dev/null
command -v systemd-analyze >/dev/null

systemd_quote() {
    local value=$1
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        printf 'Paths cannot contain newlines.\n' >&2
        return 1
    fi
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//%/%%}
    printf '"%s"' "$value"
}

WORKING_DIR=$(systemd_quote "$REPO_DIR")
PYTHON_COMMAND=$(systemd_quote "$PYTHON_BIN")
PYTHON_COMMAND=${PYTHON_COMMAND//\$/\$\$}
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

cat > "$TEMP_DIR/kilt-viewer.service" <<EOF
[Unit]
Description=KILT read-only dataset viewer
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$WORKING_DIR
ExecStart=$PYTHON_COMMAND -m kilt_synmes.viewer --host $VIEWER_HOST --port $VIEWER_PORT
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
EOF

systemd-analyze verify "$TEMP_DIR/kilt-viewer.service"
install -m 0644 "$TEMP_DIR/kilt-viewer.service" /etc/systemd/system/kilt-viewer.service
systemctl daemon-reload
systemctl enable kilt-viewer.service
systemctl restart kilt-viewer.service
systemctl --no-pager --full status kilt-viewer.service
printf '\nDataset viewer listening on %s:%s\n' "$VIEWER_HOST" "$VIEWER_PORT"
printf 'Manage: sudo systemctl {start|restart|stop|status} kilt-viewer\n'