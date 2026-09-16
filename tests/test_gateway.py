"""Tests for the resident Autonomics gateway integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench_engine.solvers.gateway import GatewayError, virtual_path


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



if __name__ == "__main__":
    unittest.main()
