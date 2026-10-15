import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_platform.anysearch import _classify, keyword_to_symbols
from quant_platform.anysearch_credentials import anysearch_token_status, save_anysearch_token


class ClassifyTests(unittest.TestCase):
    def test_classify_shanghai(self):
        self.assertEqual(_classify("600519"), "600519.SH")
        self.assertEqual(_classify("510300"), "510300.SH")
        self.assertEqual(_classify("688981"), "688981.SH")

    def test_classify_shenzhen(self):
        self.assertEqual(_classify("000001"), "000001.SZ")
        self.assertEqual(_classify("300750"), "300750.SZ")
        self.assertEqual(_classify("159915"), "159915.SZ")

    def test_classify_invalid(self):
        self.assertIsNone(_classify("123"))
        self.assertIsNone(_classify("abcdef"))
        self.assertIsNone(_classify("1234567"))


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class KeywordToSymbolsTests(unittest.TestCase):
    def test_extracts_and_dedups_symbols(self):
        payload = {
            "code": 0,
            "message": "success",
            "request_id": "r",
            "data": {
                "results": [
                    {"title": "宁德时代", "snippet": "300750.SZ 电池龙头", "content": "宁德时代(300750) 是动力电池龙头"},
                    {"title": "贵州茅台", "snippet": "600519 白酒", "content": "贵州茅台 600519.SH"},
                ],
                "metadata": {"total_results": 2, "search_time_ms": 100},
            },
        }
        with patch("quant_platform.anysearch.requests.post", return_value=FakeResponse(payload)), patch("quant_platform.anysearch._stock_lookup", return_value=({}, None)):
            symbols = keyword_to_symbols("新能源")
        self.assertIn("300750.SZ", symbols)
        self.assertIn("600519.SH", symbols)
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_error_code_raises(self):
        payload = {"code": -1, "message": "Query is required.", "request_id": "x"}
        with patch("quant_platform.anysearch.requests.post", return_value=FakeResponse(payload)):
            with self.assertRaisesRegex(RuntimeError, "Query is required"):
                keyword_to_symbols("x")

    def test_empty_keyword_raises(self):
        with self.assertRaisesRegex(ValueError, "keyword"):
            keyword_to_symbols("   ")

    def test_maps_company_names_to_codes(self):
        payload = {
            "code": 0, "message": "success", "request_id": "r",
            "data": {
                "results": [
                    {"title": "人工智能龙头", "content": "科大讯飞和宁德时代是人工智能龙头"},
                ],
                "metadata": {"total_results": 1, "search_time_ms": 10},
            },
        }
        lookup = ({"科大讯飞": "002230.SZ", "宁德时代": "300750.SZ", "贵州茅台": "600519.SH"}, {"002230.SZ", "300750.SZ", "600519.SH"})
        with patch("quant_platform.anysearch.requests.post", return_value=FakeResponse(payload)), patch("quant_platform.anysearch._stock_lookup", return_value=lookup):
            symbols = keyword_to_symbols("人工智能")
        self.assertIn("002230.SZ", symbols)
        self.assertIn("300750.SZ", symbols)
        self.assertNotIn("600519.SH", symbols)

    def test_filters_invalid_codes(self):
        payload = {
            "code": 0, "message": "success", "request_id": "r",
            "data": {
                "results": [{"title": "x", "content": "代码 283445 和 600519"}],
                "metadata": {"total_results": 1, "search_time_ms": 10},
            },
        }
        lookup = ({}, {"600519.SH"})
        with patch("quant_platform.anysearch.requests.post", return_value=FakeResponse(payload)), patch("quant_platform.anysearch._stock_lookup", return_value=lookup):
            symbols = keyword_to_symbols("x")
        self.assertEqual(symbols, ["600519.SH"])


class AnysearchCredentialsTests(unittest.TestCase):
    def test_save_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = save_anysearch_token("sk-anysearch-test-12345", root)
            self.assertTrue(result["configured"])
            self.assertTrue(anysearch_token_status(root)["configured"])

    def test_invalid_token_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "invalid"):
                save_anysearch_token("short", Path(directory))


if __name__ == "__main__":
    unittest.main()
