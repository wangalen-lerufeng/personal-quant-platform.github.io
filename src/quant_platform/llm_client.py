"""Minimal OpenAI-compatible chat client for research-only model calls."""
import json
import os
import ssl
from urllib.request import Request, urlopen

import certifi


def _secure_urlopen(request, timeout):
    return urlopen(request, timeout=timeout, context=ssl.create_default_context(cafile=certifi.where()))


class OpenAICompatibleChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str, temperature: float = 0.2, max_tokens: int = 2000, timeout: float = 90.0, opener=_secure_urlopen):
        if not base_url or not model:
            raise ValueError("base_url and model are required")
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = (os.getenv(api_key_env) or "").strip().strip('"').strip("'")
        if self.api_key and not self.api_key.isascii():
            raise ValueError("AI API key must contain ASCII characters only; remove Chinese text, Markdown, and full-width punctuation")
        if self.api_key and not self.api_key.startswith("sk-"):
            raise ValueError("AI API key format is invalid; it should start with sk-")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._opener = opener

    def complete(self, user_prompt: str) -> str:
        if not self.api_key:
            raise ValueError("AI model API key is not configured")
        system = (
            "你是个人量化平台的研究助手。只做行情、回测、组合、因子、市场状态、订单特征和风险分析；"
            "禁止下单、撤单、修改风控或索取账户密钥。需要计算时，可输出一个或多个 JSON 工具调用（最多15个），每个格式为"
            '{"tool":"工具名","arguments":{...}}。可用只读工具：'
            "get_market_data（provider/symbol/start_date/end_date/dataset/asset_class）；"
            "所有行情、基础资料和财务报表都必须使用get_market_data；禁止创造get_fundamental_data等新工具名；"
            "A股深度研究首次调用应一次列出全部所需数据集；不得传递fields、limit、offset、report_type或period等未声明参数；"
            "run_backtest（provider/symbol/start_date/end_date/dataset/asset_class/lookback/quantity）；"
            "analyze_factor（records/groups，records 每项含 factor 和 forward_return）；"
            "analyze_factor_panel（records/return_columns/groups/out_of_sample_start/dollar_volume_column，"
            "records 每项含 timestamp、symbol、factor 和明确的 forward_return 标签）；"
            "classify_market_state（bars/window，bars 每项含 close）；"
            "monitor_simulation_risk（initial_equity/equity_series/max_daily_loss/max_drawdown）；"
            "analyze_order_features（trades/large_trade_threshold，trades 每项含 side/price/quantity）。"
            "A股股票免费日线使用 provider=tushare、dataset=daily、asset_class=equity；"
            "A股ETF使用 provider=tushare、dataset=fund_daily、asset_class=etf，但可能需要额外积分；"
            "EURUSD使用本地CSV或MT5；不要为EURUSD调用Eastmoney。"
            "没有工具真实数据时不得给出虚构数值，并明确说明结论不是投资建议。"
            "财务报表原始值通常为年初至报告期末累计值，不得直接称为单季度值。"
            "单季度财务值、现金流质量、ROE/ROA和减值观察只采用程序生成的fundamental_analysis结果。"
            "每项数值结论必须标注[dataset=数据集, end_date=报告期, field=字段名]；"
            "原因解释必须引用公告、新闻或明确数据字段，否则写明‘原因未验证’。"
            "禁止仅凭assets_impair_loss正负号推断减值计提或转回。"
        )
        return self.complete_messages([{"role": "system", "content": system}, {"role": "user", "content": user_prompt}])

    def complete_messages(self, messages: list[dict]) -> str:
        if not self.api_key:
            raise ValueError("AI model API key is not configured")
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "thinking": {"type": "disabled"}, "messages": messages}
        request = Request(self.endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}, method="POST")
        with self._opener(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        try:
            message = payload["choices"][0]["message"]
            content = message.get("content") or ""
            if not content and message.get("reasoning_content"):
                content = message["reasoning_content"]
            if not content:
                raise ValueError("empty chat completion content")
            return content
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("invalid chat completions response") from error

    def complete_with_tool_result(self, user_prompt: str, assistant_answer: str, tool_result: dict) -> str:
        messages = [
            {"role": "system", "content": "你是个人量化平台的研究助手。只做行情、回测、组合和风险分析；禁止下单、撤单、修改风控或索取账户密钥。只能引用工具返回的真实数据；如果工具报错或数据为空，明确说明无法得出数值结论，不得估算或编造。所有行情、基础资料和财务报表查询都只能使用get_market_data，禁止创造get_fundamental_data等新工具名。财务报表原始值通常是累计值，单季度值、现金流质量、ROE/ROA与减值观察必须采用fundamental_analysis程序结果。每项结论标注[dataset=数据集, end_date=报告期, field=字段名]。原因解释必须关联公告、新闻或明确字段证据；缺少证据时必须写‘原因未验证’。禁止根据assets_impair_loss正负号自行推断计提或转回。A股ETF使用Tushare，EURUSD不要调用Eastmoney。"},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_answer},
            {"role": "user", "content": "研究工具执行结果（唯一可信数据源）：" + json.dumps(tool_result, ensure_ascii=False)},
        ]
        return self.complete_messages(messages)
