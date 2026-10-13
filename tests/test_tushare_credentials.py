import os
from pathlib import Path
import tempfile
import unittest

from quant_platform.tushare_credentials import save_tushare_token, tushare_token_status


class TushareCredentialTests(unittest.TestCase):
    def test_token_is_saved_to_owner_only_ignored_secret_file_without_echo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = save_tushare_token("a" * 40, root)
            secret_file = root / "secrets" / "tushare.env"

            self.assertTrue(secret_file.exists())
            self.assertEqual(secret_file.stat().st_mode & 0o777, 0o600)
            self.assertEqual(secret_file.parent.stat().st_mode & 0o777, 0o700)
            self.assertNotIn("a" * 40, str(result))
            self.assertTrue(result["configured"])

    def test_invalid_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "format"):
                save_tushare_token("bad token; export X=1", Path(directory))

    def test_status_never_returns_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_tushare_token("b" * 40, root)
            result = tushare_token_status(root)
            self.assertEqual(result["token"], "[configured]")
            self.assertNotIn("b" * 40, str(result))


if __name__ == "__main__":
    unittest.main()
