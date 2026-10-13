# Futures Auto-Rollover Implementation Plan

> **For implementer:** Use TDD throughout. Write failing test first. Watch it fail. Then implement.

**Goal:** Add an opt-in, auditable futures position rollover model to the event-driven research backtest.

**Architecture:** `MarketExecutionPolicy` remains the source of rollover rules and exposes whether automatic rollover is enabled. `BacktestEngine` reacts only when a valid successor-contract market event arrives after the rule's effective time: it closes the expiring position at its last observed price and opens the successor with equal quantity at the successor event price. Every rollover records prices, quantity, costs, timestamp and contract mapping; default behavior remains the existing conservative block.

**Tech Stack:** Python 3.11, dataclasses, unittest.

---

### Task 1: Automatic rollover domain behavior

**Files:**
- Modify: `src/quant_platform/backtest.py`
- Modify: `tests/test_backtest.py`

**Steps:**
1. Add a failing test proving default policy still blocks expired contracts.
2. Add a failing test proving opt-in rollover closes `IF2601`, opens `IF2602` with equal quantity, charges costs and records one audit event.
3. Implement the minimum policy lookup and engine transition.
4. Run `python3 -m unittest discover -s tests -p 'test_backtest.py'` and the full suite.

### Task 2: HTTP configuration and evidence

**Files:**
- Modify: `src/quant_platform/http_api.py`
- Modify: `tests/test_http_api.py`

**Steps:**
1. Add a failing API payload test for `auto_rollover: true`.
2. Pass the flag into `MarketExecutionPolicy` and expose `rollovers` in the result.
3. Run HTTP tests and the full suite.

### Task 3: Documentation and runtime verification

**Files:**
- Modify: `README.md`
- Modify: `docs/roadmap.md`

**Steps:**
1. Document opt-in behavior, required successor prices and paper-only boundary.
2. Run compilation, diff checks and all tests.
3. Restart the local service and verify `/health` remains paper-only.
