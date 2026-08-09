#!/usr/bin/env python3
"""Install a built Game Detector bundle for the current Linux user."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


APP_ID = "game-detector"
EXECUTABLE_NAME = "GameDetector"
INSTALL_MARKER = ".game-detector-install"


def _default_bundle() -> Path:
    script_path = Path(__file__).resolve()
    repository_bundle = script_path.parents[1] / "dist" / "GameDetector"
    packaged_bundle = script_path.parent.parent
    if (repository_bundle / EXECUTABLE_NAME).is_file():
        return repository_bundle
    return packaged_bundle


def _data_home() -> Path:
    configured = os.environ.get("XDG_DATA_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share"


def _find_setup_file(bundle: Path, relative_name: str, repository_name: str) -> Path:
    candidates = (
        bundle / "setup" / relative_name,
        Path(__file__).resolve().parents[1] / repository_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Installation resource not found: {relative_name}")


def _safe_existing_install(path: Path) -> bool:
    return path.is_dir() and (path / INSTALL_MARKER).is_file()


def _quote_desktop_exec(path: Path) -> str:
    # The freedesktop Exec format uses double quotes and backslash escaping.
    value = str(path)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{escaped}"'


def _write_atomic(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def install(bundle: Path) -> Path:
    bundle = bundle.expanduser().resolve()
    executable = bundle / EXECUTABLE_NAME
    if not executable.is_file():
        raise FileNotFoundError(f"Built application not found: {executable}")

    data_home = _data_home()
    install_dir = data_home / APP_ID
    previous_dir = data_home / f"{APP_ID}.previous"
    applications_dir = data_home / "applications"
    icon_dir = data_home / "icons" / "hicolor" / "scalable" / "apps"

    desktop_template_path = _find_setup_file(
        bundle,
        "game-detector.desktop.in",
        "packaging/linux/game-detector.desktop.in",
    )
    icon_source = _find_setup_file(
        bundle,
        "game-detector.svg",
        "assets/game-detector.svg",
    )

    data_home.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{APP_ID}.new-", dir=data_home))
    try:
        # PyInstaller uses relative symlinks extensively on Linux. Preserve
        # them so the installed app stays compact and keeps the layout that its
        # native-library loader was tested against.
        shutil.copytree(bundle, staging_dir, dirs_exist_ok=True, symlinks=True)
        (staging_dir / INSTALL_MARKER).write_text(
            "Managed by Game Detector's per-user installer.\n",
            encoding="utf-8",
        )

        if install_dir.exists() and not _safe_existing_install(install_dir):
            raise RuntimeError(
                f"Refusing to replace unmanaged path: {install_dir}. "
                "Move it aside and run the installer again."
            )
        if previous_dir.exists() and not _safe_existing_install(previous_dir):
            raise RuntimeError(
                f"Refusing to replace unmanaged backup path: {previous_dir}."
            )

        if previous_dir.exists():
            shutil.rmtree(previous_dir)
        if install_dir.exists():
            os.replace(install_dir, previous_dir)
        os.replace(staging_dir, install_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    installed_executable = install_dir / EXECUTABLE_NAME
    desktop_template = desktop_template_path.read_text(encoding="utf-8")
    desktop_content = desktop_template.replace(
        "@EXECUTABLE@",
        _quote_desktop_exec(installed_executable),
    )
    _write_atomic(
        applications_dir / f"{APP_ID}.desktop",
        desktop_content,
        0o644,
    )
    _write_atomic(
        icon_dir / f"{APP_ID}.svg",
        icon_source.read_text(encoding="utf-8"),
        0o644,
    )

    desktop_database = shutil.which("update-desktop-database")
    if desktop_database:
        subprocess.run(
            [desktop_database, str(applications_dir)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    for cache_builder_name in ("kbuildsycoca6", "kbuildsycoca5"):
        cache_builder = shutil.which(cache_builder_name)
        if cache_builder:
            subprocess.run(
                [cache_builder, "--noincremental"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            break
    return installed_executable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the Game Detector app for the current Linux user."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=_default_bundle(),
        help="Path to the built one-folder GameDetector bundle.",
    )
    return parser


def main() -> int:
    if not sys.platform.startswith("linux"):
        print("This installer is only for Linux.", file=sys.stderr)
        return 2
    try:
        installed_executable = install(build_parser().parse_args().bundle)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        return 2

    print("Game Detector is installed for this user.")
    print("Open your application menu and click Game Detector.")
    print(f"Installed executable: {installed_executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
