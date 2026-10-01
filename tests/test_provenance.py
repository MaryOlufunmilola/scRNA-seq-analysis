"""
Unit tests for scripts/provenance.py.
"""

import hashlib

import pytest

import provenance


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"GSE203612" * 1000)
    assert provenance.sha256_file(str(path), chunk_size=7) == hashlib.sha256(b"GSE203612" * 1000).hexdigest()


def test_verify_checksum_records_when_not_pinned(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    assert provenance.verify_checksum(str(path), None, "test file") == hashlib.sha256(b"abc").hexdigest()


def test_verify_checksum_accepts_match_case_insensitively(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest().upper()
    assert provenance.verify_checksum(str(path), expected, "test file") == expected.lower()


def test_verify_checksum_mismatch_raises_when_strict(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="checksum mismatch"):
        provenance.verify_checksum(str(path), "0" * 64, "test file", strict=True)


def test_verify_checksum_mismatch_warns_when_not_strict(tmp_path, capsys):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    digest = provenance.verify_checksum(str(path), "0" * 64, "test file", strict=False)
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert "Warning" in capsys.readouterr().out


def test_package_versions_reports_missing_packages():
    versions = provenance.package_versions(["pytest", "definitely-not-a-real-package-xyz"])
    assert "python" in versions
    assert versions["pytest"] != "not installed"
    assert versions["definitely-not-a-real-package-xyz"] == "not installed"
