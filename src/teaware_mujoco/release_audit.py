from __future__ import annotations

import re
import subprocess
from pathlib import Path

TEXT_SUFFIXES = {
    "",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
FORBIDDEN_TRACKED_PREFIXES = ("data/", "output/", "checkpoints/", ".env")
SENSITIVE_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "macOS user path": re.compile("/" + r"Users/[^/\s]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "hardware serial port": re.compile(r"/dev/(?:tty|cu)\.[A-Za-z0-9._-]+|/dev/ttyUSB\d+"),
    "private network address": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
}


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode() for item in result.stdout.split(b"\0") if item]


def audit_public_release(root: str | Path) -> list[str]:
    repo_root = Path(root).expanduser().resolve()
    issues: list[str] = []
    required_notices = (
        repo_root / "THIRD_PARTY_NOTICES.md",
        repo_root / "src/teaware_mujoco/assets/ufactory_xarm7/LICENSE",
    )
    for path in required_notices:
        if not path.is_file():
            issues.append(f"missing required license/notice: {path.relative_to(repo_root)}")

    for path in tracked_files(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        if relative.startswith(FORBIDDEN_TRACKED_PREFIXES):
            issues.append(f"runtime or sensitive path is tracked: {relative}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(content):
                issues.append(f"{relative}: possible {label}")
    return sorted(set(issues))
