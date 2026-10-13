"""Deterministic normalization and calculations for financial-statement research."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


INCOME_CUMULATIVE_FIELDS = (
    "revenue", "total_revenue", "operate_profit", "total_profit", "n_income",
    "n_income_attr_p", "deductprofit", "basic_eps", "diluted_eps", "ebit",
    "ebitda", "sell_exp", "admin_exp", "fin_exp", "rd_exp", "assets_impair_loss",
)
CASHFLOW_CUMULATIVE_FIELDS = (
    "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
    "c_fr_sale_sg", "c_pay_acq_const_fiolta", "n_incr_cash_cash_equ",
)
VERIFIED_INDICATOR_FIELDS = (
    "roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets",
    "current_ratio", "quick_ratio",
)


def _date(value: Any) -> date | None:
    text = str(value or "").replace("-", "")
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError:
        return None


def _evidence(dataset: str, row: dict, field: str) -> dict:
    return {
        "dataset": dataset,
        "end_date": row.get("end_date"),
        "field": field,
        "ann_date": row.get("ann_date"),
    }


def _normalize(rows: list[dict], as_of: date) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("end_date"):
            continue
        announcement = _date(row.get("ann_date"))
        if announcement and announcement > as_of:
            continue
        grouped.setdefault(str(row["end_date"]), []).append(dict(row))

    normalized = []
    for period_rows in grouped.values():
        normalized.append(max(period_rows, key=lambda row: (
            _date(row.get("ann_date")) or date.min,
            str(row.get("update_flag", "")),
        )))
    return sorted(normalized, key=lambda row: str(row["end_date"]))


def _single_quarters(dataset: str, rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    output: list[dict] = []
    previous_by_year: dict[str, dict] = {}
    for row in rows:
        period = str(row["end_date"])
        year, month_day = period[:4], period[4:]
        previous = previous_by_year.get(year)
        item = {"end_date": period, "ann_date": row.get("ann_date"), "calculation": "standalone_quarter"}
        item["evidence"] = {}
        for field in fields:
            current = row.get(field)
            if not isinstance(current, (int, float)):
                continue
            prior = previous.get(field) if previous else None
            if month_day == "0331" or not isinstance(prior, (int, float)):
                item[field] = current
                item["evidence"][field] = [_evidence(dataset, row, field)]
            else:
                item[field] = current - prior
                item["evidence"][field] = [
                    _evidence(dataset, row, field), _evidence(dataset, previous, field)
                ]
        output.append(item)
        previous_by_year[year] = row
    return output


def analyze_fundamental_datasets(tool_results: list[dict], as_of_date: date | None = None) -> dict:
    """Return auditable calculations without making causal or accounting-sign claims."""
    as_of = as_of_date or date.today()
    raw: dict[str, list[dict]] = {}
    for result in tool_results:
        dataset, data = result.get("dataset"), result.get("data")
        if isinstance(dataset, str) and isinstance(data, list):
            raw.setdefault(dataset, []).extend(row for row in data if isinstance(row, dict))

    normalized = {dataset: _normalize(rows, as_of) for dataset, rows in raw.items()}
    income = normalized.get("income", [])
    cashflow = normalized.get("cashflow", [])
    indicators = normalized.get("fina_indicator", [])

    single_quarter = {
        "income": _single_quarters("income", income, INCOME_CUMULATIVE_FIELDS),
        "cashflow": _single_quarters("cashflow", cashflow, CASHFLOW_CUMULATIVE_FIELDS),
    }

    income_by_period = {str(row["end_date"]): row for row in income}
    cash_flow_quality = []
    for cash_row in cashflow:
        income_row = income_by_period.get(str(cash_row["end_date"]))
        numerator = cash_row.get("n_cashflow_act")
        denominator = income_row.get("n_income_attr_p") if income_row else None
        if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)) and denominator != 0:
            cash_flow_quality.append({
                "end_date": cash_row["end_date"],
                "value": numerator / denominator,
                "formula": "cashflow.n_cashflow_act / income.n_income_attr_p",
                "evidence": [
                    _evidence("cashflow", cash_row, "n_cashflow_act"),
                    _evidence("income", income_row, "n_income_attr_p"),
                ],
            })

    impairment = []
    for row in income:
        value = row.get("assets_impair_loss")
        if isinstance(value, (int, float)):
            impairment.append({
                "end_date": row["end_date"], "raw_value": value,
                "interpretation": "provider_sign_only_no_reversal_inference",
                "evidence": _evidence("income", row, "assets_impair_loss"),
            })

    verified = []
    for row in indicators:
        item = {"end_date": row["end_date"], "ann_date": row.get("ann_date"), "evidence": {}}
        for field in VERIFIED_INDICATOR_FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)):
                item[field] = value
                item["evidence"][field] = _evidence("fina_indicator", row, field)
        if item["evidence"]:
            verified.append(item)

    return {
        "as_of_date": as_of.isoformat(),
        "normalized": normalized,
        "single_quarter": single_quarter,
        "computed_metrics": {"cash_flow_quality": cash_flow_quality},
        "impairment_observations": impairment,
        "verified_indicators": verified,
        "interpretation_policy": "programmatic_values_only; causal_claims_require_external_or_field_evidence",
    }
