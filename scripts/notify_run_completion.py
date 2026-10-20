#!/usr/bin/env python3
"""监控组合研究跑测结束，并向指定邮箱发送结果摘要。

用法:
    python3 scripts/notify_run_completion.py [--timeout-minutes 60]

读取 ``secrets/qqmail.env`` 中的 ``QQMAIL_AUTH_CODE``，通过 QQ 邮箱 SMTP
（smtp.qq.com:465 SSL）发信到 ``wang.alen@qq.com``。仅在检测到新的
phase9 跑测记录（相对脚本启动时的最新记录）后发送一次。
"""

import argparse
import json
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_HISTORY = ROOT / "data" / "run_history.json"
SECRET_FILE = ROOT / "secrets" / "qqmail.env"

SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
FROM_ADDR = "wang.alen@qq.com"
TO_ADDR = "wang.alen@qq.com"


def read_auth_code() -> str:
    for line in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export ") and "QQMAIL_AUTH_CODE=" in line:
            return line.split("=", 1)[1].strip()
    raise RuntimeError("未在 secrets/qqmail.env 找到 QQMAIL_AUTH_CODE")


def _load_runs():
    data = json.loads(RUN_HISTORY.read_text(encoding="utf-8"))
    runs = data if isinstance(data, list) else data.get("runs", data.get("records", []))
    return runs if isinstance(runs, list) else list(runs.values())


def latest_phase9_run_id():
    ids = [str(r.get("run_id") or "") for r in _load_runs()]
    phase9 = [i for i in ids if i.startswith("phase9-")]
    return max(phase9) if phase9 else None


def summarize(run: dict) -> str:
    detail = run.get("detail") or {}
    gate = detail.get("model_gate") or {}
    metrics = (detail.get("charts") or {}).get("metrics") or {}
    targets = run.get("targets") or []
    target_text = ", ".join(
        f"{t.get('symbol')}:{float(t.get('weight', 0)) * 100:.1f}%" for t in targets
    ) or "无"
    lines = [
        f"run_id: {run.get('run_id')}",
        f"资产类型: {run.get('asset_class')} · 指数: {run.get('index_name') or '手动'}",
        f"区间: {run.get('start_date')} ~ {run.get('end_date')} · top_k={run.get('top_k')}",
        f"LightGBM Rank IC: {run.get('baseline_rank_ic')}",
        f"DoubleEnsemble Rank IC: {run.get('ensemble_rank_ic')}",
        f"模型准入: {'通过' if gate.get('passed') else '失败'}"
        + (f"（{gate.get('reason')}）" if gate.get('reason') else ""),
        f"候选池: {', '.join(run.get('candidates') or []) or '无'}",
        f"目标仓位: {target_text}",
        f"扣费后净收益: {metrics.get('net_return_after_cost')}",
        f"仿真账户: {run.get('account_id')}",
    ]
    return "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    import certifi

    auth = read_auth_code()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = FROM_ADDR
    msg["To"] = TO_ADDR
    msg.set_content(body)
    context = ssl.create_default_context(cafile=certifi.where())
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(FROM_ADDR, auth)
        server.send_message(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-minutes", type=int, default=60)
    args = parser.parse_args()

    baseline = latest_phase9_run_id()
    deadline = time.time() + args.timeout_minutes * 60
    while time.time() < deadline:
        current = latest_phase9_run_id()
        if current and current != baseline:
            run = next(r for r in _load_runs() if str(r.get("run_id")) == current)
            subject = f"组合研究跑测已完成：{current}"
            send_email(subject, summarize(run))
            print(f"EMAIL_SENT {current}")
            return 0
        time.sleep(5)
    print("TIMEOUT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
