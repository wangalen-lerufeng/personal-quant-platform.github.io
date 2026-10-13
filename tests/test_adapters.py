import unittest

from quant_platform.adapters import CtpPaperGateway, Mt5PaperGateway, QmtPaperGateway
from quant_platform.domain import OrderIntent, OrderSide


class AdapterTests(unittest.TestCase):
    def test_all_market_gateways_share_submit_contract(self):
        for gateway_cls in (QmtPaperGateway, CtpPaperGateway, Mt5PaperGateway):
            with self.subTest(gateway=gateway_cls.__name__):
                receipt = gateway_cls().submit_order(
                    OrderIntent(symbol="TEST", side=OrderSide.BUY, quantity=1, strategy_id="contract")
                )
                self.assertEqual(receipt.status, "accepted")
                self.assertEqual(receipt.symbol, "TEST")

    def test_all_paper_gateways_report_execution_disabled(self):
        for gateway_cls in (QmtPaperGateway, CtpPaperGateway, Mt5PaperGateway):
            with self.subTest(gateway=gateway_cls.__name__):
                self.assertFalse(gateway_cls().health_check()["execution_enabled"])

    def test_paper_gateway_rejects_unknown_cancel(self):
        with self.assertRaises(KeyError):
            QmtPaperGateway().cancel_order("unknown-order")
