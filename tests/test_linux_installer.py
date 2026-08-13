from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import install_linux_desktop as installer


class LinuxInstallerTests(unittest.TestCase):
    def test_uninstall_removes_only_managed_paths_and_menu_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            for name in ("proaim", "proaim.previous"):
                target = data / name
                target.mkdir()
                (target / installer.INSTALL_MARKER).write_text("managed\n")
            desktop = data / "applications" / "proaim.desktop"
            icon = data / "icons/hicolor/scalable/apps/proaim.svg"
            desktop.parent.mkdir(parents=True)
            icon.parent.mkdir(parents=True)
            desktop.write_text("entry")
            icon.write_text("icon")
            with mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(data)}), mock.patch(
                "scripts.install_linux_desktop._refresh_desktop"
            ):
                removed = installer.uninstall()
            self.assertEqual(len(removed), 4)
            self.assertFalse((data / "proaim").exists())
            self.assertFalse((data / "proaim.previous").exists())

    def test_uninstall_refuses_unmanaged_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "proaim").mkdir()
            with mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(data)}):
                with self.assertRaisesRegex(RuntimeError, "unmanaged"):
                    installer.uninstall()


if __name__ == "__main__":
    unittest.main()
