"""Fail-closed provenance helpers for manifests, workers, resume, and aggregation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import locale
import os
import platform
import re
import subprocess
import sys
from typing import Iterable


class ProvenanceMismatch(ValueError):
    """A raw result does not belong to the manifest row that names it."""


# Frozen manifests predate ``source_commit`` fields.  Register their exact bytes and observed
# execution checkpoints so legacy validation is explicit rather than permissive.
FROZEN_ROUND2_FINAL_MANIFEST_SHA256 = (
    "53acb6c62b0a24f1ff7dc236244f9ca6c287e767c8ed52c1ccc596cf5162820d"
)
FROZEN_ROUND2_DEVELOPMENT_MANIFEST_SHA256 = (
    "513ff579cc3d748c9d00c0a9655d0c649027c7b7aa778f3ca6ac47155706717f"
)
LEGACY_MANIFEST_COMMITS = {
    FROZEN_ROUND2_FINAL_MANIFEST_SHA256: "aa983a2",
    FROZEN_ROUND2_DEVELOPMENT_MANIFEST_SHA256: "6122aeb",
}
LEGACY_MANIFEST_PHASES = {
    FROZEN_ROUND2_FINAL_MANIFEST_SHA256: "final",
    FROZEN_ROUND2_DEVELOPMENT_MANIFEST_SHA256: "development",
}


def registered_manifest_commit(manifest_sha256: str) -> str | None:
    return LEGACY_MANIFEST_COMMITS.get(manifest_sha256)


def registered_manifest_phase(manifest_sha256: str) -> str | None:
    return LEGACY_MANIFEST_PHASES.get(manifest_sha256)


def paths_overlap(first: str, second: str) -> bool:
    """Return whether two real paths are equal or one contains the other."""
    first_real, second_real = os.path.realpath(first), os.path.realpath(second)
    try:
        common = os.path.commonpath([first_real, second_real])
    except ValueError:
        return False
    return common in {first_real, second_real}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result_checksum(path: str, *, required: bool) -> None:
    checksum_path = path + ".sha256"
    if not os.path.exists(checksum_path):
        if required:
            raise ProvenanceMismatch(f"missing checksum for {path}")
        return
    with open(checksum_path, encoding="ascii") as handle:
        expected = handle.read().strip()
    actual = sha256_file(path)
    if expected != actual:
        raise ProvenanceMismatch(
            f"checksum mismatch for {path}: expected {expected!r}, got {actual!r}"
        )


def canonical_json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dirty_paths_from_porcelain(status: str) -> list[str]:
    """Extract paths from ``git status --porcelain=v1`` output.

    Round-2 runtime paths contain no whitespace or rename arrows.  Handling the rename form
    as two dirty paths still makes the check fail closed for any source rename.
    """
    paths: list[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        payload = line[3:].strip()
        paths.extend(payload.split(" -> ") if " -> " in payload else [payload])
    return paths


def _paths_content_sha256(root: str, paths: Iterable[str]) -> str:
    """Hash current bytes for dirty paths so same-path edits cannot evade end checks."""
    digest = hashlib.sha256()
    for relative in sorted(set(paths)):
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        path = os.path.join(root, relative)
        if os.path.islink(path):
            digest.update(b"symlink\0" + os.readlink(path).encode("utf-8"))
        elif os.path.isfile(path):
            digest.update(b"file\0")
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif os.path.isdir(path):
            digest.update(b"directory\0")
        else:
            digest.update(b"missing\0")
        digest.update(b"\0")
    return digest.hexdigest()


def git_state(root: str, *, allowed_dirty_paths: Iterable[str] = ()) -> dict:
    """Capture the source commit and fail-closed source-dirty state.

    A generated manifest is deliberately allowed to differ from HEAD: its bytes are protected
    separately by ``manifest_sha256`` and it must name the already committed source revision.
    No code, configuration, or other uncommitted path is exempted.
    """
    try:
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return {
            "commit": "nogit",
            "dirty": True,
            "worktree_dirty": True,
            "dirty_paths": ["<git-state-unavailable>"],
            "allowed_dirty_paths": [],
            "status_sha256": None,
            "worktree_content_sha256": None,
        }
    allowed = {
        os.path.relpath(path, root).replace(os.sep, "/") if os.path.isabs(path)
        else os.path.normpath(path).replace(os.sep, "/")
        for path in allowed_dirty_paths
    }
    dirty_paths = _dirty_paths_from_porcelain(status)
    unexpected_dirty_paths = [path for path in dirty_paths if path not in allowed]
    return {
        "commit": commit,
        "dirty": bool(unexpected_dirty_paths),
        "worktree_dirty": bool(status),
        "dirty_paths": unexpected_dirty_paths,
        "allowed_dirty_paths": [path for path in dirty_paths if path in allowed],
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "worktree_content_sha256": _paths_content_sha256(root, dirty_paths),
    }


def environment_snapshot() -> dict:
    """Record the scientific stack plus loaded numerical-runtime details."""
    distributions = (
        "torch", "numpy", "pandas", "scikit-learn", "econml", "scipy",
        "matplotlib", "PyYAML", "pyarrow", "pytest",
    )
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu_model = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            cpu_model = next(
                line.split(":", 1)[1].strip()
                for line in handle
                if line.lower().startswith("model name")
            )
    except (OSError, StopIteration, IndexError):
        pass
    out = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "locale": locale.setlocale(locale.LC_ALL, None),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "packages": versions,
    }
    np = sys.modules.get("numpy")
    if np is not None:
        try:
            dependencies = np.__config__.show(mode="dicts").get("Build Dependencies", {})
            out["blas_lapack"] = {
                name: {
                    key: value for key, value in details.items()
                    if key in {"name", "found", "version", "openblas configuration"}
                }
                for name, details in dependencies.items()
                if name in {"blas", "lapack"}
            }
        except Exception as exc:
            out["blas_lapack_error"] = f"{type(exc).__name__}: {exc}"
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            out["torch_runtime"] = {
                "cuda_build": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_devices": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ],
                "cudnn_version": torch.backends.cudnn.version(),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "num_threads": int(torch.get_num_threads()),
                "num_interop_threads": int(torch.get_num_interop_threads()),
            }
        except Exception as exc:
            out["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    return out


def validate_manifest_rows(rows: Iterable[dict]) -> None:
    rows = list(rows)
    paths = [row.get("out_path") for row in rows]
    keys = [(row.get("study"), row.get("config_id"), row.get("scientific_seed")) for row in rows]
    for label, values in (("out_path", paths), ("task key", keys)):
        if len(values) != len(set(values)):
            raise ProvenanceMismatch(f"manifest contains duplicate {label}")
    for seed_name in ("bootstrap_seed", "bank_seed"):
        values = [row.get(seed_name) for row in rows]
        if len(values) != len(set(values)):
            raise ProvenanceMismatch(f"manifest contains duplicate {seed_name}")
    has_source = [bool(row.get("source_commit")) for row in rows]
    if any(has_source) and not all(has_source):
        raise ProvenanceMismatch("manifest mixes registered and missing source commits")
    source_commits = {row.get("source_commit") for row in rows if row.get("source_commit")}
    if len(source_commits) > 1:
        raise ProvenanceMismatch("manifest mixes multiple source commits")
    if all(has_source) and rows:
        clean_policies = {row.get("require_clean_source") for row in rows}
        if clean_policies - {True, False} or len(clean_policies) != 1:
            raise ProvenanceMismatch("manifest has missing or mixed source-cleanliness policies")


def validate_execution_manifest(rows: Iterable[dict]) -> None:
    """Require a new executable manifest to pin one full commit and clean policy."""
    rows = list(rows)
    validate_manifest_rows(rows)
    if not rows:
        raise ProvenanceMismatch("execution manifest is empty")
    for row in rows:
        source_commit = row.get("source_commit")
        if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise ProvenanceMismatch(
                "execution manifest requires a full 40-character lowercase-hex source_commit"
            )
        if not isinstance(row.get("require_clean_source"), bool):
            raise ProvenanceMismatch(
                "execution manifest requires an explicit boolean source-cleanliness policy"
            )


def validate_manifest_output_namespace(
    rows: Iterable[dict], *, protected_roots: Iterable[str],
) -> None:
    """Reject manifest outputs that resolve inside evidence-preservation namespaces."""
    protected_roots = tuple(protected_roots)
    for row in rows:
        out_path = row.get("out_path")
        if not isinstance(out_path, str) or not out_path or not os.path.isabs(out_path):
            raise ProvenanceMismatch("manifest out_path must be a nonempty absolute path")
        for protected_root in protected_roots:
            if paths_overlap(out_path, protected_root):
                raise ProvenanceMismatch(
                    f"manifest targets protected output namespace {protected_root}: {out_path}"
                )


def _compare(mismatches: list[str], label: str, actual, expected) -> None:
    if actual != expected:
        mismatches.append(f"{label}: expected {expected!r}, got {actual!r}")


def validate_result_provenance(
    prov: dict,
    expected: dict,
    *,
    array_task_id: int,
    manifest_sha256: str,
    authorization_commit: str | None = None,
    release_authorization_sha256: str | None = None,
) -> None:
    """Raise if a raw record is stale, swapped, mis-seeded, or from the wrong source."""
    mismatches: list[str] = []
    _compare(mismatches, "config", prov.get("config"), expected)
    _compare(mismatches, "array_task_id", prov.get("array_task_id"), int(array_task_id))
    _compare(mismatches, "study", prov.get("experiment"), expected.get("study"))
    _compare(mismatches, "dispatch", prov.get("dispatch"), expected.get("dispatch"))
    for key in ("scientific_seed", "bootstrap_seed", "split_seed", "bank_seed"):
        _compare(mismatches, key, prov.get(key), expected.get(key))

    source_commit = expected.get("source_commit")
    if source_commit:
        release_path = expected.get("release_authorization_path")
        if release_path:
            _compare(
                mismatches,
                "scientific_source_commit",
                prov.get("scientific_source_commit"),
                source_commit,
            )
            if not isinstance(authorization_commit, str) or not re.fullmatch(
                r"[0-9a-f]{40}", authorization_commit,
            ):
                mismatches.append("expected authorization_commit is missing or invalid")
            else:
                _compare(
                    mismatches,
                    "recorded authorization_commit",
                    prov.get("authorization_commit"),
                    authorization_commit,
                )
                _compare(
                    mismatches,
                    "execution authorization commit",
                    prov.get("git_commit_full"),
                    authorization_commit,
                )
            if not isinstance(release_authorization_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", release_authorization_sha256,
            ):
                mismatches.append("expected release authorization hash is missing or invalid")
            else:
                _compare(
                    mismatches,
                    "release_authorization_sha256",
                    prov.get("release_authorization_sha256"),
                    release_authorization_sha256,
                )
            if prov.get("git_worktree_dirty") is not False:
                mismatches.append("released worker worktree was not fully clean")
        else:
            _compare(mismatches, "source_commit", prov.get("git_commit_full"), source_commit)
        _compare(mismatches, "manifest_sha256", prov.get("manifest_sha256"), manifest_sha256)
        if expected.get("require_clean_source", True) and prov.get("git_dirty") is not False:
            mismatches.append("worker source was dirty but manifest requires a clean source")
    else:
        legacy_commit = registered_manifest_commit(manifest_sha256)
        if legacy_commit is None:
            mismatches.append("manifest has no source_commit and is not a registered frozen manifest")
        else:
            recorded_commit = prov.get("git_commit_full") or prov.get("git_commit")
            if recorded_commit != legacy_commit:
                mismatches.append(
                    f"legacy checkpoint: expected {legacy_commit!r}, got {recorded_commit!r}"
                )

    status = prov.get("status")
    if status == "ok":
        _compare(mismatches, "exit_code", prov.get("exit_code"), 0)
        result = prov.get("result")
        if not isinstance(result, dict):
            mismatches.append("successful record has no result object")
        else:
            if "study" not in result:
                mismatches.append("successful result is missing study")
            else:
                _compare(mismatches, "result.study", result.get("study"), expected.get("study"))
            if "seed" not in result:
                mismatches.append("successful result is missing seed")
            else:
                _compare(
                    mismatches,
                    "result.seed",
                    result.get("seed"),
                    expected.get("scientific_seed"),
                )
    elif status == "error":
        if not isinstance(prov.get("exit_code"), int) or prov.get("exit_code") == 0:
            mismatches.append("error record must have a nonzero integer exit_code")
        if not isinstance(prov.get("error"), str) or not prov.get("error").strip():
            mismatches.append("error record must have a nonempty error message")
        if prov.get("failure_kind") not in {"algorithmic", "infrastructure", "provenance"}:
            mismatches.append("error record has an invalid failure_kind")
    elif status != "ok":
        mismatches.append(f"invalid status {status!r}")

    if mismatches:
        path = expected.get("out_path", "<unknown>")
        raise ProvenanceMismatch(f"provenance mismatch for {path}: " + "; ".join(mismatches))
