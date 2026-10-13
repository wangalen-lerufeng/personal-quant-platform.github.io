from .app import build_paper_app
from .config import RuntimeConfig


def run_check() -> dict[str, object]:
    app = build_paper_app()
    return {
        "status": "ok",
        "environment": app.config.environment,
        "live_trading_enabled": app.config.live_trading_enabled,
        "ai_execution_tools_enabled": app.config.ai_execution_tools_enabled,
        "execution_enabled": app.config.live_trading_enabled or app.config.ai_execution_tools_enabled,
        "paper_only": not app.config.live_trading_enabled and not app.config.ai_execution_tools_enabled,
        "adapters": app.control_plane.adapter_health(),
    }


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Personal Quant Platform")
    parser.add_argument("command", choices=("check", "serve"), nargs="?", default="check")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", default=None, help="YAML runtime configuration file")
    args = parser.parse_args(argv)
    config = RuntimeConfig.from_file(args.config) if args.config else RuntimeConfig()
    app = build_paper_app(config=config)
    if args.command == "check":
        print(json.dumps(run_check(), ensure_ascii=False, indent=2))
        return 0
    from .http_api import serve
    serve(
        app.config,
        host=args.host,
        port=args.port,
        control_plane=app.control_plane,
        research_tools=app.research_tools,
        simulation_registry=app.simulation_registry,
    )
    return 0
