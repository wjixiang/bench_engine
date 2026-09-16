"""Tests for the resident Autonomics gateway integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench_engine.solvers.gateway import GatewayError, ensure_headless_holder, virtual_path


class GatewayHelperTest(unittest.TestCase):
    def test_virtual_path_maps_under_gateway_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            root = state / "gateway-root"
            root.mkdir()
            (state / "vfs.toml").write_text(
                "[[backend]]\n"
                'id = "default"\n'
                'type = "local"\n'
                'root = "/"\n\n'
                "[[mount]]\n"
                'path = "/"\n'
                'backend = "default"\n'
                f'source = "{root}"\n'
                "read_only = false\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ", {"AUTONOMICS_STATE_DIR": str(state)}, clear=True
            ):
                self.assertEqual(virtual_path(root), "/")
                self.assertEqual(
                    virtual_path(root / "runs" / "task-1"),
                    "/runs/task-1",
                )
                with self.assertRaisesRegex(
                    GatewayError, "under the gateway VFS root"
                ):
                    virtual_path(state / "outside")

                linked_data = root / "task" / "data"
                linked_data.parent.mkdir()
                linked_data.symlink_to(state / "external-data")
                self.assertEqual(virtual_path(linked_data), "/task/data")

    def test_holder_is_not_spawned_when_headless_is_live(self) -> None:
        state = {
            "active_model_spec": "provider:active-model",
            "profiles": [{"path": "researcher"}],
            "agents": [{"path": "/root/headless"}],
        }
        requests: list[tuple[str, str, object]] = []

        def fake_request(method: str, path: str, *, payload: object = None) -> object:
            requests.append((method, path, payload))
            return state

        with (
            patch("bench_engine.solvers.gateway._ensure_gateway"),
            patch("bench_engine.solvers.gateway._request", side_effect=fake_request),
        ):
            ensure_headless_holder(
                Path("/usr/bin/autonomics"),
                profile=None,
                model=None,
            )

        self.assertEqual(requests, [("GET", "/state", None)])

    def test_holder_is_spawned_when_headless_is_absent(self) -> None:
        profile = {"path": "researcher", "runtime": {}}
        state = {
            "active_model_spec": "provider:active-model",
            "profiles": [profile],
            "agents": [],
        }
        requests: list[tuple[str, str, object]] = []

        def fake_request(method: str, path: str, *, payload: object = None) -> object:
            requests.append((method, path, payload))
            if method == "GET":
                return state
            return {"path": "/root/headless"}

        with (
            patch("bench_engine.solvers.gateway._ensure_gateway"),
            patch("bench_engine.solvers.gateway._request", side_effect=fake_request),
        ):
            ensure_headless_holder(
                Path("/usr/bin/autonomics"),
                profile=Path("researcher"),
                model=None,
            )

        self.assertEqual(requests[0], ("GET", "/state", None))
        self.assertEqual(requests[1][0], "POST")
        self.assertEqual(requests[1][1], "/agents")
        spawn_payload = requests[1][2]
        assert isinstance(spawn_payload, dict)
        self.assertEqual(spawn_payload["name"], "headless")
        self.assertEqual(spawn_payload["parent_path"], "/root")
        self.assertEqual(spawn_payload["profile"], profile)
        self.assertEqual(spawn_payload["model_spec"], "provider:active-model")


if __name__ == "__main__":
    unittest.main()
