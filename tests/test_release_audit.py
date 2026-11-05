"""Dependency-free regressions for citation handling in the public audit."""

import tempfile
import unittest
from pathlib import Path

from scripts.audit_release import audit

# Assemble obsolete names so this regression fixture also passes the source audit.
AUTHOR = "Li" + "u"
TITLE = "Learning to Fly from Point Clouds via Differentiable Simulation"
CITATIONS = {
    "README.md": f"- {AUTHOR} et al. (IROS 2026) — *{TITLE}*.",
    "docs/pointnet.md": f"Source: {AUTHOR} et al., *{TITLE}*, equations 5–7.",
}


class ReleaseAuditCitationTests(unittest.TestCase):
    def findings(self, name, text):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return audit(root)

    def test_recognized_citations_are_allowed(self):
        for name, citation in CITATIONS.items():
            with self.subTest(name=name):
                self.assertEqual(self.findings(name, citation + " More attribution text."), [])

    def test_actual_documentation_passes(self):
        root = Path(__file__).resolve().parents[1]
        for name in CITATIONS:
            with self.subTest(name=name):
                self.assertEqual(self.findings(name, (root / name).read_text(encoding="utf-8")), [])

    def test_old_identifiers_still_fail_beside_citations(self):
        identifiers = (AUTHOR, AUTHOR.lower() + "_policy", AUTHOR.upper() + "-config", "Zh" + "ang", "diffphys" + "drone")
        for name, citation in CITATIONS.items():
            for identifier in identifiers:
                for separator in (" ", "\n"):
                    with self.subTest(name=name, identifier=identifier, separator=separator):
                        findings = self.findings(name, citation + separator + identifier)
                        self.assertTrue(any("obsolete project or personal identifier" in item for item in findings))

    def test_citations_are_scoped_to_the_document_and_paper(self):
        for name, citation in CITATIONS.items():
            for target, text in (
                ("docs/other.md", citation),
                ("src/policy.py", citation),
                (name, citation.replace(TITLE, "A different paper")),
                (name, citation.replace(AUTHOR, AUTHOR + "_policy")),
            ):
                with self.subTest(target=target, text=text):
                    self.assertTrue(self.findings(target, text))

    def test_citations_do_not_hide_private_paths_or_networks(self):
        for name, citation in CITATIONS.items():
            for value, marker in (
                ("/home/" + "localuser/private", "machine-specific path"),
                ("100.92." + "1.2", "internal network address"),
            ):
                with self.subTest(name=name, value=value):
                    self.assertTrue(any(marker in item for item in self.findings(name, citation + " " + value)))

    def test_obsolete_file_names_still_fail(self):
        self.assertTrue(self.findings(f"src/{AUTHOR.lower()}_policy.py", "# policy\n"))


if __name__ == "__main__":
    unittest.main()
