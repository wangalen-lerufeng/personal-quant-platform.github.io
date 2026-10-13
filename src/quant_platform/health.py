def health_snapshot(environment: str, live_trading_enabled: bool, ai_execution_tools_enabled: bool = False) -> dict[str, object]:
    return {
        "status": "ok",
        "environment": environment,
        "live_trading_enabled": live_trading_enabled,
        "ai_execution_tools_enabled": ai_execution_tools_enabled,
        "execution_enabled": live_trading_enabled or ai_execution_tools_enabled,
        "paper_only": not live_trading_enabled and not ai_execution_tools_enabled,
    }
