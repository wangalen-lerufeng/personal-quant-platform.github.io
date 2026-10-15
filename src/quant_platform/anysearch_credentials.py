"""Owner-only local persistence for the AnySearch API key."""

import os
from pathlib import Path
import re
import tempfile


_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._-]{16,256}$")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def save_anysearch_token(token: str, project_root: Path | None = None) -> dict:
    normalized = str(token or "").strip()
    if not _TOKEN_PATTERN.fullmatch(normalized):
        raise ValueError("AnySearch token format is invalid")
    root = Path(project_root) if project_root is not None else _project_root()
    secret_dir = root / "secrets"
    secret_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    secret_dir.chmod(0o700)
    target = secret_dir / "anysearch.env"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".anysearch-", dir=secret_dir)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"export ANYSEARCH_TOKEN={normalized}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        target.chmod(0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise
    os.environ["ANYSEARCH_TOKEN"] = normalized
    return {
        "configured": True,
        "token": "[configured]",
        "storage": "owner_only_local_file",
        "restart_required": False,
        "execution_enabled": False,
    }


def anysearch_token_status(project_root: Path | None = None) -> dict:
    root = Path(project_root) if project_root is not None else _project_root()
    stored = (root / "secrets" / "anysearch.env").is_file()
    configured = stored or bool((os.getenv("ANYSEARCH_TOKEN") or "").strip())
    return {
        "configured": configured,
        "token": "[configured]" if configured else "[not configured]",
        "storage": "owner_only_local_file" if stored else "environment",
        "execution_enabled": False,
    }
