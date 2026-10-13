from pathlib import Path
import re
import unittest


class DependencyLockTests(unittest.TestCase):
    def test_runtime_dependencies_are_exactly_pinned(self):
        root = Path(__file__).resolve().parents[1]
        lock_lines = [
            line.strip()
            for line in (root / "requirements.lock").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertGreaterEqual(len(lock_lines), 6)
        for line in lock_lines:
            self.assertRegex(line, r"^[A-Za-z0-9_.-]+==[^=<>~!]+$")
            self.assertNotRegex(line, re.compile(r"https?://", re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
