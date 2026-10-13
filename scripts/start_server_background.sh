#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
LABEL="com.personal.quant-platform"
SOURCE_PLIST="${PROJECT_DIR}/scripts/${LABEL}.plist"
AGENT_DIR="${HOME}/Library/LaunchAgents"
AGENT_PLIST="${AGENT_DIR}/${LABEL}.plist"
DOMAIN="gui/${UID}"
LOG_FILE="${PROJECT_DIR}/logs/server.log"

mkdir -p "${PROJECT_DIR}/logs" "${AGENT_DIR}"
chmod 700 "${PROJECT_DIR}/logs"
touch "${LOG_FILE}"
chmod 600 "${LOG_FILE}"

existing_pid="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "${existing_pid}" ]]; then
  echo "服务已经运行：PID=${existing_pid}，地址=http://${HOST}:${PORT}/"
  exit 0
fi

if [[ ! -f "${SOURCE_PLIST}" ]]; then
  echo "缺少 launchd 配置：${SOURCE_PLIST}" >&2
  exit 1
fi

cp "${SOURCE_PLIST}" "${AGENT_PLIST}"
launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
launchctl bootstrap "${DOMAIN}" "${AGENT_PLIST}"
launchctl kickstart -k "${DOMAIN}/${LABEL}"

sleep 2
server_pid="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "${server_pid}" ]]; then
  echo "服务已由 macOS 后台托管：PID=${server_pid}"
  echo "访问地址：http://${HOST}:${PORT}/"
  echo "健康检查：http://${HOST}:${PORT}/health"
  echo "日志文件：${LOG_FILE}"
else
  echo "服务启动失败，请查看：${LOG_FILE}" >&2
  exit 1
fi
