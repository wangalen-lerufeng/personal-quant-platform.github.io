#!/bin/zsh
set -eo pipefail
umask 077

PROJECT_DIR="/Users/apple/personal-quant-platform"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
CONFIG="${CONFIG:-${PROJECT_DIR}/configs/settings.example.yaml}"

if [[ -f "${HOME}/.zshrc" ]]; then
  source "${HOME}/.zshrc"
fi
if [[ -f "${PROJECT_DIR}/secrets/tushare.env" ]]; then
  source "${PROJECT_DIR}/secrets/tushare.env"
fi

# Project-specific names prevent shell profiles from replacing the safe
# loopback binding with a machine hostname.
QP_SERVICE_HOST="${QP_SERVICE_HOST:-127.0.0.1}"
QP_SERVICE_PORT="${QP_SERVICE_PORT:-8080}"
# Always use the official Tushare SDK endpoint for the desktop service.
unset TUSHARE_BASE_URL

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export QUANT_PLATFORM_LAUNCHD=1
cd "${PROJECT_DIR}"
exec arch -arm64 "${PYTHON_BIN}" -m quant_platform serve \
  --config "${CONFIG}" \
  --host "${QP_SERVICE_HOST}" \
  --port "${QP_SERVICE_PORT}"
