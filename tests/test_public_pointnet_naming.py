"""Allow historical checkpoint tokens without allowing them as new source labels."""

from aerialnav.pointnet.config import CHECKPOINT_METHODS, OBSERVATION_CONTRACTS
from scripts.audit_release import audit


def test_checkpoint_metadata_exception_does_not_hide_new_personal_identifiers(tmp_path):
    folder = tmp_path / "src/aerialnav/pointnet"
    folder.mkdir(parents=True)
    (folder / "config.py").write_text(
        f"CHECKPOINT_METHODS = {CHECKPOINT_METHODS!r}\nOBSERVATION_CONTRACTS = {OBSERVATION_CONTRACTS!r}\n"
    )
    assert audit(tmp_path) == []
    (folder / "model.py").write_text(f"method = {CHECKPOINT_METHODS[1]!r}\n")
    findings = audit(tmp_path)
    assert len(findings) == 1 and "model.py:1" in findings[0]
