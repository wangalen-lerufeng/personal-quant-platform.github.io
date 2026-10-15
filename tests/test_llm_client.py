import json
import os
import unittest
from quant_platform.llm_client import OpenAICompatibleChatClient


class Response:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def read(self): return json.dumps(self.payload).encode()


class LlmClientTests(unittest.TestCase):
    def test_chat_completion_uses_openai_compatible_request(self):
        captured = {}
        def opener(request, timeout):
            captured["url"] = request.full_url; captured["auth"] = request.get_header("Authorization"); captured["body"] = json.loads(request.data)
            return Response({"choices": [{"message": {"content": "研究结论"}}]})
        os.environ["TEST_AI_KEY"] = "sk-test-secret"
        try:
            answer = OpenAICompatibleChatClient("https://api.deepseek.com", "deepseek-flash", "TEST_AI_KEY", opener=opener).complete("分析回测")
        finally:
            os.environ.pop("TEST_AI_KEY", None)
        self.assertEqual(answer, "研究结论")
        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["auth"], "Bearer sk-test-secret")
        self.assertEqual(captured["body"]["model"], "deepseek-flash")
        self.assertNotIn("thinking", captured["body"])
        system_prompt = captured["body"]["messages"][0]["content"]
        self.assertIn("禁止下单", system_prompt)
        self.assertIn("analyze_factor", system_prompt)
        self.assertIn("classify_market_state", system_prompt)
        self.assertIn("monitor_simulation_risk", system_prompt)
        self.assertIn("analyze_order_features", system_prompt)
        self.assertIn("fundamental_analysis", system_prompt)
        self.assertIn("原因未验证", system_prompt)
        self.assertIn("assets_impair_loss", system_prompt)

    def test_error_payload_is_surfaced(self):
        def opener(request, timeout):
            return Response({"error": {"message": "Model Not Exist", "code": 30001}})
        os.environ["TEST_AI_KEY"] = "sk-test-secret"
        try:
            client = OpenAICompatibleChatClient("https://api.deepseek.com", "deepseek-flash", "TEST_AI_KEY", opener=opener)
            with self.assertRaisesRegex(ValueError, "Model Not Exist"):
                client.complete("分析")
        finally:
            os.environ.pop("TEST_AI_KEY", None)

    def test_follow_up_requires_field_level_financial_evidence(self):
        captured = {}
        def opener(request, timeout):
            captured["body"] = json.loads(request.data)
            return Response({"choices": [{"message": {"content": "核验结论"}}]})
        os.environ["TEST_AI_KEY"] = "sk-test-secret"
        try:
            client = OpenAICompatibleChatClient("https://api.deepseek.com", "deepseek-flash", "TEST_AI_KEY", opener=opener)
            client.complete_with_tool_result("分析", "调用工具", {"results": []})
        finally:
            os.environ.pop("TEST_AI_KEY", None)
        system_prompt = captured["body"]["messages"][0]["content"]
        self.assertIn("[dataset=数据集, end_date=报告期, field=字段名]", system_prompt)
        self.assertIn("原因未验证", system_prompt)

    def test_missing_key_is_rejected(self):
        os.environ.pop("MISSING_AI_KEY", None)
        with self.assertRaises(ValueError):
            OpenAICompatibleChatClient("https://api.test", "model", "MISSING_AI_KEY").complete("hello")

    def test_non_ascii_key_is_rejected_before_http_request(self):
        os.environ["BAD_AI_KEY"] = "你的 DeepSeek Key"
        try:
            with self.assertRaisesRegex(ValueError, "ASCII"):
                OpenAICompatibleChatClient("https://api.test", "model", "BAD_AI_KEY")
        finally:
            os.environ.pop("BAD_AI_KEY", None)
