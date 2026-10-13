from dataclasses import dataclass
from pathlib import Path

from .adapter_registry import AdapterRegistry
from .adapters import CtpAdapter, Mt5Adapter, QmtAdapter
from .config import RuntimeConfig
from .control import ControlPlane
from .experiments import SqliteExperimentStore
from .research_tools import ResearchTools
from .risk import RiskEngine, RiskLimits
from .simulation import PaperSimulationRegistry
from .storage import SimulationAccountStore, SqliteAuditStore
from .portfolio import PositionBook
from .market_data import InMemoryMarketDataFeed
from .strategy_registry import StrategyRegistry


@dataclass
class PaperApp:
    config: RuntimeConfig
    control_plane: ControlPlane
    research_tools: ResearchTools
    audit_store: SqliteAuditStore
    experiment_store: SqliteExperimentStore
    market_data_feed: InMemoryMarketDataFeed
    simulation_registry: PaperSimulationRegistry


def build_paper_app(database: str | None = None, config: RuntimeConfig | None = None, risk_limits: RiskLimits | None = None) -> PaperApp:
    config = config or RuntimeConfig()
    if config.live_trading_enabled or config.ai_execution_tools_enabled:
        raise ValueError("paper app cannot enable live trading or AI execution tools")
    database = config.database_path if database is None else database
    if database != ":memory:":
        database_parent = Path(database).expanduser().parent
        database_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        database_parent.chmod(0o700)
    audit_store = SqliteAuditStore(database)
    experiment_store = SqliteExperimentStore(database)
    risk_engine = RiskEngine(risk_limits or RiskLimits())
    adapters = AdapterRegistry([QmtAdapter(), CtpAdapter(), Mt5Adapter()])
    market_data_feed = InMemoryMarketDataFeed()
    strategy_registry = StrategyRegistry([("momentum", {"status": "research-only", "execution_enabled": False})])
    control_plane = ControlPlane(
        config, risk_engine, audit_store, adapters, experiment_store, PositionBook(), strategy_registry, market_data_feed
    )
    research_tools = ResearchTools(control_plane, experiment_store, market_data_feed)
    simulation_registry = PaperSimulationRegistry(SimulationAccountStore(database))
    return PaperApp(
        config,
        control_plane,
        research_tools,
        audit_store,
        experiment_store,
        market_data_feed,
        simulation_registry,
    )
