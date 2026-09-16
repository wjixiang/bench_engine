"""Helpers for routing Autonomics runs through the resident gateway."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_PREFIX = "/api/v1"
DEFAULT_ADDR = "127.0.0.1:8765"
HEADLESS_AGENT_PATH = "/root/headless"


class GatewayError(RuntimeError):
    """Raised when the gateway cannot support isolated benchmark runs."""


def _state_dir() -> Path:
    configured = os.environ.get("AUTONOMICS_STATE_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".autonomics"


def _base_url() -> str:
    addr = os.environ.get("AUTONOMICS_HTTP_API_ADDR", DEFAULT_ADDR)
    if addr.startswith(("http://", "https://")):
        return addr.rstrip("/")
    return f"http://{addr}".rstrip("/")


def _token() -> str:
    configured = os.environ.get("AUTONOMICS_HTTP_API_TOKEN")
    if configured:
        return configured
    token_path = _state_dir() / "gateway.token"
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise GatewayError(f"cannot read gateway token {token_path}: {exc}") from exc


def _request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _base_url() + API_PREFIX + path,
        data=body,
        method=method,
        headers={"Authorization": f"Bearer {_token()}"},
    )
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise GatewayError(f"gateway request failed ({method} {path}): {exc}") from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GatewayError(f"gateway returned invalid JSON for {method} {path}") from exc


def _ensure_gateway(executable: Path) -> None:
    command = [str(executable), "run", "--list-sessions", "--json"]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = (
            exc.stderr
            if isinstance(exc, subprocess.CalledProcessError)
            else ""
        )
        suffix = f": {detail.strip()}" if detail.strip() else ""
        raise GatewayError(f"cannot reach Autonomics gateway{suffix}") from exc


def _selected_profile(state: dict[str, Any], profile: Path | None) -> dict[str, Any]:
    profiles = state.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise GatewayError("gateway has no agent profiles configured")
    if profile is None:
        first = profiles[0]
        if not isinstance(first, dict):
            raise GatewayError("gateway returned an invalid agent profile")
        return first
    requested = str(profile)
    for entry in profiles:
        if isinstance(entry, dict) and entry.get("path") == requested:
            return entry
    raise GatewayError(f"gateway does not have profile {requested!r}")


def _agent_is_live(state: dict[str, Any], agent_path: str) -> bool:
    agents = state.get("agents", [])
    return any(
        isinstance(entry, dict) and entry.get("path") == agent_path
        for entry in agents
    )


def _spawn_headless_holder(
    state: dict[str, Any],
    profile: Path | None,
    model: str | None,
) -> None:
    if _agent_is_live(state, HEADLESS_AGENT_PATH):
        return
    payload: dict[str, Any] = {
        "name": "headless",
        "parent_path": "/root",
        "profile": _selected_profile(state, profile),
        "model_spec": model or state.get("active_model_spec"),
    }
    _request("POST", "/agents", payload=payload)


def ensure_headless_holder(
    executable: Path,
    *,
    profile: Path | None,
    model: str | None,
) -> None:
    """Keep the stable headless identity live as a contention holder.

    Gateway runs fall back to a unique ``headless-xxxxxxxx`` agent identity when
    ``/root/headless`` is already live. Those fallback identities have no stored
    history, so each benchmark task starts with a fresh auto-created session.
    """
    _ensure_gateway(executable)
    state = _request("GET", "/state")
    if not isinstance(state, dict):
        raise GatewayError("gateway state is not an object")
    if _agent_is_live(state, HEADLESS_AGENT_PATH):
        return
    try:
        _spawn_headless_holder(state, profile, model)
    except GatewayError:
        # A concurrently-updated gateway may report the agent as absent just
        # before another client spawns it. Re-read state before failing.
        updated = _request("GET", "/state")
        if not isinstance(updated, dict) or not _agent_is_live(
            updated, HEADLESS_AGENT_PATH
        ):
            raise


def _gateway_vfs_root() -> Path:
    manifest_path = _state_dir() / "vfs.toml"
    try:
        with manifest_path.open("rb") as handle:
            manifest = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise GatewayError(f"cannot read gateway VFS manifest {manifest_path}: {exc}") from exc
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
