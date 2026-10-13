from .control import ControlPlane


class ResearchTools:
    """Read-only tools intended for a future AI Agent integration."""

    _CATALOG = {
        "system_status": "读取系统与交易模式",
        "risk_status": "读取当前风控限制",
        "recent_orders": "读取最近订单审计记录",
        "audit_events": "读取追加式订单状态事件",
        "experiment_results": "读取回测实验结果",
        "market_data": "只读查询标准化行情",
        "portfolio_state": "读取只读组合估值快照",
        "strategies": "读取已登记策略元数据",
        "run_backtest": "运行纸面历史回测并记录实验结果",
        "analyze_order_features": "计算订单流不平衡、VWAP、大单与价格冲击",
        "analyze_factor": "计算因子 IC、Rank IC 与分组收益",
        "analyze_factor_panel": "计算因子 IC 序列、ICIR、衰减、换手率、容量与样本外表现",
        "classify_market_state": "以可解释规则识别趋势、震荡与高波动状态",
        "monitor_simulation_risk": "回放仿真权益并检查亏损、回撤与熔断阈值",
    }

    def __init__(self, control_plane: ControlPlane, experiment_store=None, market_data_feed=None):
        self.control_plane = control_plane
        self.experiment_store = experiment_store
        self.market_data_feed = market_data_feed

    def catalog(self) -> dict[str, str]:
        return dict(self._CATALOG)

    def system_status(self) -> dict[str, object]:
        return self.control_plane.system_state()

    def risk_status(self) -> dict[str, object]:
        return self.control_plane.risk_state()

    def recent_orders(self) -> list[dict]:
        return self.control_plane.orders()

    def audit_events(self, order_id: str | None = None) -> list[dict]:
        return self.control_plane.audit_events(order_id)

    def experiment_results(self) -> list[dict]:
        if self.experiment_store is None:
            return []
        return self.experiment_store.list()

    def market_data(self, symbol: str | None = None, limit: int | None = None) -> list[dict]:
        if self.market_data_feed is None:
            return []
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        events = self.market_data_feed.events
        if symbol is not None:
            events = [event for event in events if event.symbol == symbol]
        rows = [
            {
                "symbol": event.symbol,
                "asset_class": event.asset_class.value,
                "price": event.price,
                "timestamp": event.timestamp.isoformat(),
            }
            for event in events
        ]
        return rows[-limit:] if limit is not None else rows

    def portfolio_state(self, prices: dict[str, float] | None = None) -> dict[str, object]:
        """Return a read-only portfolio snapshot for research and monitoring."""
        if prices is None and self.market_data_feed is not None:
            prices = self.market_data_feed.latest_prices()
        return self.control_plane.portfolio_state(prices)

    def strategies(self) -> list[dict]:
        return self.control_plane.strategies()

    def run_backtest(self, strategy_id: str, strategy, events, commission_rate: float = 0.0, slippage_rate: float = 0.0):
        from .backtest import BacktestEngine

        result = BacktestEngine(
            strategy,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        ).run(events)
        if self.experiment_store is not None:
            metrics = result.to_dict()
            metrics.update({"commission_rate": commission_rate, "slippage_rate": slippage_rate})
            self.experiment_store.save(strategy_id, metrics)
        return result
