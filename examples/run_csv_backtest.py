"""Run a local backtest from a standard CSV file."""

import sys

from quant_platform.backtest import BacktestEngine
from quant_platform.market_data import CsvMarketDataLoader
from quant_platform.strategy import MomentumStrategy


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a CSV backtest")
    parser.add_argument("path")
    parser.add_argument("--commission-rate", type=float, default=0.0)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    events = CsvMarketDataLoader().load(args.path)
    result = BacktestEngine(
        MomentumStrategy(lookback=2, quantity=0.01),
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
    ).run(events)
    print({
        "signals": result.signal_count,
        "trades": result.trade_count,
        "total_return": result.total_return,
        "win_rate": result.win_rate,
        "max_drawdown": result.max_drawdown,
        "commission_rate": args.commission_rate,
        "slippage_rate": args.slippage_rate,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
