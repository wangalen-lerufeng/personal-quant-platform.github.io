import unittest

from quant_platform.config import RuntimeConfig
from quant_platform.control import ControlPlane
from quant_platform.experiments import ExperimentStore
from quant_platform.http_api import api_payload
from quant_platform.risk import RiskEngine, RiskLimits
from quant_platform.storage import SqliteAuditStore


class ExperimentApiTests(unittest.TestCase):
    def test_experiment_route_returns_saved_results(self):
        experiments = ExperimentStore()
        experiments.save("trend-v1", {"total_return": 0.12})
        control = ControlPlane(RuntimeConfig(), RiskEngine(RiskLimits()), SqliteAuditStore(":memory:"), experiment_store=experiments)

        payload = api_payload("/experiments", RuntimeConfig(), control)

        self.assertEqual(payload["experiments"][0]["strategy_id"], "trend-v1")
        self.assertFalse(payload["execution_enabled"])
