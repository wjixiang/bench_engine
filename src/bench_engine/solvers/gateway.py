"""Helpers for mapping host paths into the resident Autonomics gateway VFS."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


class GatewayError(RuntimeError):
    """Raised when a host path cannot be represented in the gateway VFS."""


def _state_dir() -> Path:
    configured = os.environ.get("AUTONOMICS_STATE_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".autonomics"


def _gateway_vfs_root() -> Path:
    manifest_path = _state_dir() / "vfs.toml"
    try:
        with manifest_path.open("rb") as handle:
            manifest: dict[str, Any] = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise GatewayError(
            f"cannot read gateway VFS manifest {manifest_path}: {exc}"
        ) from exc

    for mount in manifest.get("mount", []):
        if (
            isinstance(mount, dict)
            and mount.get("path") == "/"
            and isinstance(mount.get("source"), str)
        ):
            return Path(mount["source"]).expanduser().resolve()
    raise GatewayError("gateway VFS has no writable root mount")


def virtual_path(host_path: Path) -> str:
    """Map a host path under the gateway VFS root to its virtual path."""
    root = _gateway_vfs_root()
    # Do not resolve symlinks: a benchmark task's mounted ``data`` link may
    # point to source storage outside the gateway root, while the gateway sees
    # the link itself below its writable root.
    host = host_path.expanduser().absolute()
    try:
        relative = host.relative_to(root)
    except ValueError as exc:
        raise GatewayError(
            "gateway mode requires --data-mount-path under the gateway VFS root "
            f"({root})"
        ) from exc
    return "/" if not relative.parts else "/" + relative.as_posix()
