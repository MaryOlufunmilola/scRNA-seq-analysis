"""Provenance helpers: file checksums and installed package versions."""

import hashlib
import platform
from importlib import metadata


def sha256_file(path, chunk_size=1 << 20):
    """SHA-256 hex digest of a file, read in chunks (safe for multi-GB archives)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path, expected, description, strict=True):
    """Compute the file's SHA-256 and compare it with `expected`."""
    actual = sha256_file(path)
    if expected is None:
        print(f"{description} SHA-256: {actual} (not pinned yet -- record this value to pin it).")
    elif actual.lower() != expected.lower():
        message = (
            f"{description} checksum mismatch: expected {expected}, got {actual}. "
            f"The file at {path} differs from the one these results were produced with."
        )
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}")
    else:
        print(f"{description} SHA-256 verified: {actual}")
    return actual


def package_versions(names):
    """{package: version} for the named distributions ('not installed' if absent)."""
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
