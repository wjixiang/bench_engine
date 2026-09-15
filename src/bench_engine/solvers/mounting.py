"""Per-task input-data mounting for solver backends."""

from __future__ import annotations

from pathlib import Path

from bench_engine.core.interfaces import Example


def _validate_component(value: str, field: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"unsafe {field}: {value!r}")


def mount_example_data(
    example: Example,
    data_mount_path: Path | None,
) -> Path | None:
    """Mount an example's assets below a task-specific mount directory."""
    if data_mount_path is None:
        return None

    _validate_component(example.id, "task id")
    task_mount_path = data_mount_path / example.id
    task_mount_path.mkdir(parents=True, exist_ok=True)

    for asset in example.assets:
        _validate_component(asset.name, "asset name")
        relative_path = Path(asset.relative_path or asset.name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe asset path: {relative_path}")
        destination = task_mount_path / relative_path
        source = asset.path.resolve()
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"data mount destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=source.is_dir())

    return task_mount_path
