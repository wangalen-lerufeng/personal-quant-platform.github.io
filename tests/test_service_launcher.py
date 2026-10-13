import unittest
from pathlib import Path


class ServiceLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.background = (root / "scripts" / "start_server_background.sh").read_text(encoding="utf-8")
        cls.wrapper = (root / "scripts" / "launch_server.zsh").read_text(encoding="utf-8")
        cls.plist = (root / "scripts" / "com.personal.quant-platform.plist").read_text(encoding="utf-8")
        cls.finder_launcher = (root / "启动量化平台.command").read_text(encoding="utf-8")

    def test_background_service_is_managed_by_launchd(self):
        self.assertIn("launchctl bootstrap", self.background)
        self.assertIn("launchctl kickstart", self.background)
        self.assertNotIn("nohup", self.background)
        self.assertIn("<key>KeepAlive</key>", self.plist)
        self.assertIn("<true/>", self.plist)
        self.assertIn('chmod 700 "${PROJECT_DIR}/logs"', self.background)
        self.assertIn('chmod 600 "${LOG_FILE}"', self.background)

    def test_launch_wrapper_uses_zsh_and_does_not_embed_api_keys(self):
        self.assertTrue(self.wrapper.startswith("#!/bin/zsh"))
        self.assertIn(".zshrc", self.wrapper)
        self.assertIn("exec arch -arm64", self.wrapper)
        self.assertNotIn("sk-", self.wrapper)
        self.assertNotIn("AI_API_KEY=", self.wrapper)
        self.assertNotIn('HOST="${HOST:-', self.wrapper)
        self.assertIn('QP_SERVICE_HOST="${QP_SERVICE_HOST:-127.0.0.1}"', self.wrapper)
        self.assertIn("unset TUSHARE_BASE_URL", self.wrapper)
        self.assertNotIn("118.89.117.77", self.wrapper)
        self.assertNotIn("token=", self.wrapper)
        self.assertIn("secrets/tushare.env", self.wrapper)

    def test_finder_launcher_does_not_load_zsh_config_in_bash(self):
        self.assertTrue(self.finder_launcher.startswith("#!/usr/bin/env bash"))
        self.assertNotIn('source "${HOME}/.zshrc"', self.finder_launcher)


if __name__ == "__main__":
    unittest.main()
