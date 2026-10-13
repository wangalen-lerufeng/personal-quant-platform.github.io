# Evidence-Based Fundamental Research Implementation Plan

> **For implementer:** Use TDD throughout. Write failing test first. Watch it fail. Then implement.

**Goal:** Make A-share deep research date-current, programmatically calculated, source-attributed, and explicit about model interpretation risk.

**Architecture:** Tushare remains the read-only raw-data source. A deterministic Python analyzer deduplicates disclosures, applies an as-of date, converts cumulative income/cash-flow periods into standalone quarters, computes selected ratios, and emits field-level evidence. DeepSeek receives this derived result and is limited to explaining it.

**Tech Stack:** Python 3.11, pandas, unittest, existing HTTP/HTML console and OpenAI-compatible client.

---

### Task 1: Current-date research requests

**Files:**
- Modify: `web/index.html`
- Test: `tests/test_web_ui.py`

1. Add a failing test requiring the stock prompt builder to derive today's date at runtime.
2. Run the web test and confirm failure.
3. Add an ISO current-date helper and include the as-of date in the generated prompt.
4. Run the web test and confirm success.

### Task 2: Deterministic financial normalization

**Files:**
- Create: `src/quant_platform/fundamental_analysis.py`
- Create: `tests/test_fundamental_analysis.py`

1. Add failing tests for disclosure deduplication, announcement-date filtering, cumulative-to-single-quarter conversion, cash-flow quality, and raw impairment sign handling.
2. Run the focused test and confirm failure.
3. Implement the minimum analyzer with field-level evidence.
4. Run focused and full tests.

### Task 3: Inject verified analysis into Agent evidence

**Files:**
- Modify: `src/quant_platform/http_api.py`
- Modify: `src/quant_platform/llm_client.py`
- Test: `tests/test_http_api.py`

1. Add a failing test requiring a deterministic `fundamental_analysis` result after financial tool batches.
2. Confirm failure.
3. Append deterministic analysis to tool results and instruct the model to cite dataset, report period, field, and announcement date for causal claims.
4. Run focused and full tests.

### Task 4: Output budget and evidence language

**Files:**
- Modify: `configs/settings.example.yaml`
- Modify: `web/index.html`
- Modify: `tests/test_web_ui.py`

1. Add failing assertions for a 7000-token research budget and qualified evidence wording.
2. Confirm failure.
3. Set the budget and replace the overstrong evidence badge text.
4. Run all tests and compile checks.

### Task 5: Restart and runtime verification

1. Restart the launchd-managed service.
2. Verify `/health`, model configuration, and the updated page.
3. Confirm real trading and AI execution remain disabled.
