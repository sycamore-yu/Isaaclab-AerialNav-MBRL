"""Audit the complete public source and documentation for obsolete project identifiers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def audit(root: Path) -> list[str]:
    """Report internal naming and workstation-path leaks, preserving proper upstream credits."""
    root = root.resolve()
    scanned_dirs = ("src", "scripts", "configs", "tests", "docs", ".github")
    scanned_root_files = ("README.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", ".gitignore", "pyproject.toml", "uv.lock", "CITATION.cff")
    text_extensions = {".py", ".json", ".md", ".yaml", ".yml", ".toml", ".cff"}
    obsolete_identifier = re.compile(r"(?i:isaaclab[-_]navrl|zhang|\bliu(?:[_-]|\b)|diffphysdrone)")
    legacy_project = re.compile(r"(?i:navrl)")
    workstation_path = re.compile(
        r"(?i:/home/[^/\s]+/|/Users/[^/\s]+/|[A-Z]:\\\\Users\\\\|/tongworkspace/|reference_repos/)"
    )
    internal_network = re.compile(r"\b100\.92\.\d{1,3}\.\d{1,3}\b")
    files = [root / filename for filename in scanned_root_files]
    for dirname in scanned_dirs:
        base = root / dirname
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file() and path.suffix in text_extensions)
    findings = []
    for path in sorted(set(files)):
        if not path.is_file() or path.is_symlink() or path.resolve() == Path(__file__).resolve():
            continue  # Do not match regex definitions against themselves.
        relative = path.relative_to(root)
        name = relative.as_posix()
        if obsolete_identifier.search(name):
            findings.append(f"{name}: obsolete file name")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            findings.append(f"{name}: non-text file in public text audit")
            continue
        for number, line in enumerate(lines, start=1):
            markers = []
            if obsolete_identifier.search(line):
                markers.append("obsolete project or personal identifier")
            # The origin of third-party components is a factual attribution,
            # not an internal method/package identifier.
            allowed_credit = (
                name == "THIRD_PARTY_NOTICES.md"
                or (name == "README.md" and "original NavRL teams" in line and "third-party" in line)
            )
            if legacy_project.search(line) and not allowed_credit:
                markers.append("unexpected upstream project name outside notices")
            if workstation_path.search(line):
                markers.append("machine-specific path")
            if internal_network.search(line):
                markers.append("internal network address")
            for marker in dict.fromkeys(markers):
                findings.append(f"{name}:{number}: {marker}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    findings = audit(args.root)
    if findings:
        print("\n".join(findings))
        raise SystemExit(1)
    print("Public source, documentation, path and naming audit passed.")


if __name__ == "__main__":
    main()
