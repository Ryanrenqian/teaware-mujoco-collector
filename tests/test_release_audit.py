from __future__ import annotations

from pathlib import Path

from teaware_mujoco.release_audit import audit_public_release


def test_repository_passes_public_release_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    assert audit_public_release(root) == []
