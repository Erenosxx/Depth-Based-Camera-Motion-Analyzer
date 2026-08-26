"""Public dosyalarda makineye özel mutlak yol olmamalı."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN = ("/home/" "han", "/mnt/" "data/", "/media/" "han")
_SKIP_DIRS = {
    ".git", "Result", "Data", "frames", "__pycache__",
    ".pytest_cache", ".venv", "venv",
}
_SKIP_NAMES = {"local.env"}
_TEXT_SUFFIXES = {".py", ".md", ".sh", ".toml", ".txt", ".yml", ".yaml",
                  ".example", ".gitignore", ".cfg", ".ini"}


def test_public_files_have_no_host_paths():
    hits: list[str] = []
    for path in _REPO_ROOT.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.name in _SKIP_NAMES or not path.is_file():
            continue
        if path.suffix not in _TEXT_SUFFIXES and path.name not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(_REPO_ROOT)
        for needle in _FORBIDDEN:
            if needle in text:
                hits.append(f"{rel}: {needle}")
    assert hits == [], "makine yolu sızdı:\n" + "\n".join(hits)
