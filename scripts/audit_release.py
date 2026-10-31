"""Check a proposed public source tree for legacy API names and local-only paths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def audit(root: Path) -> list[str]:
    """Return source-level findings; original upstream notices are outside this scope."""
    roots = ("src", "scripts", "configs", "tests")
    forbidden = {
        "old package or method": re.compile(r"(?i:isaaclab_navrl|zhang|navrl)"),
        "workstation path": re.compile(r"(?i:/home/[^/\s]+/|/Users/[^/\s]+/|[A-Z]:\\\\Users\\\\)"),
        "internal dashboard": re.compile(r"(?:https?://)?100\.92\.\d{1,3}\.\d{1,3}"),
    }
    findings = []
    for dirname in roots:
        for path in (root / dirname).rglob("*"):
            if path.resolve() == Path(__file__).resolve():
                continue  # Rules must not report their own pattern definitions.
            if not path.is_file() or path.suffix not in {".py", ".json", ".yaml", ".yml"}:
                continue
            for index, line in enumerate(path.read_text().splitlines(), start=1):
                for label, pattern in forbidden.items():
                    if pattern.search(line):
                        findings.append(f"{path.relative_to(root)}:{index}: {label}")
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    findings = audit(args.root)
    if findings:
        print("\n".join(findings))
        raise SystemExit(1)
    print("Public source namespace/path audit passed.")


if __name__ == "__main__":
    main()
