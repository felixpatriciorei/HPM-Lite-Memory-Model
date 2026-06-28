"""Check that README.md local image/file references exist.

This prevents pushing a GitHub front page with broken figure links.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_IMG_RE = re.compile(r"<img\s+[^>]*src=[\"']([^\"']+)[\"']", re.IGNORECASE)

IGNORE_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "#",
)


def normalize_target(raw: str) -> str:
    target = raw.strip()
    if " " in target and not target.startswith("<"):
        # Markdown can have optional title text after the URL. Keep the URL only.
        target = target.split()[0]
    target = target.strip("<>")
    target = target.split("#", 1)[0]
    return target


def main() -> int:
    if not README.exists():
        print(f"missing {README}")
        return 1

    text = README.read_text(encoding="utf-8")
    raw_targets = []
    raw_targets.extend(m.group(1) for m in MARKDOWN_LINK_RE.finditer(text))
    raw_targets.extend(m.group(1) for m in HTML_IMG_RE.finditer(text))

    missing: list[str] = []
    checked: list[str] = []
    for raw in raw_targets:
        target = normalize_target(raw)
        if not target or target.startswith(IGNORE_PREFIXES):
            continue
        # Skip pure code-ish placeholders.
        if target.startswith("{") or target.startswith("$"):
            continue
        path = (ROOT / target).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            missing.append(f"{target} (escapes repo root)")
            continue
        checked.append(target)
        if not path.exists():
            missing.append(target)

    print(f"checked {len(checked)} local README references")
    if missing:
        print("missing references:")
        for item in missing:
            print(f"  - {item}")
        return 1

    print("README local references OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
