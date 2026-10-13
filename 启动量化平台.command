#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
START_LOG="${PROJECT_DIR}/logs/finder-start.log"
mkdir -p "${PROJECT_DIR}/logs"
exec >>"${START_LOG}" 2>&1
trap 'echo; echo "启动失败，详情请查看：${START_LOG}"; echo "按回车关闭此窗口。"; read -r' ERR

cd "${PROJECT_DIR}"
./scripts/start_server_background.sh

echo
echo "量化平台已启动。"
echo "浏览器地址：http://127.0.0.1:8080/"
open "http://127.0.0.1:8080/" 2>/dev/null || true
echo "按回车关闭此窗口。"
read -r
