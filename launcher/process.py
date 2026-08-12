"""Child-process lifecycle helpers used by the Tk launcher."""

from __future__ import annotations

from collections.abc import Callable
import ntpath
import os
import signal
import shutil
import subprocess
import sys
from typing import Sequence


def _start_managed_process(command: Sequence[str]) -> subprocess.Popen[str]:
    """Start a managed child in its own process group with merged output."""

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    options: dict[str, object] = {}
    if os.name == "nt":
        options["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        options["start_new_session"] = True

    return subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
        **options,
    )


def start_detector(command: Sequence[str]) -> subprocess.Popen[str]:
    """Start the detector as a managed child process."""

    return _start_managed_process(command)


def start_precision_controller(command: Sequence[str]) -> subprocess.Popen[str]:
    """Start mapping verification or controller precision as its own child."""

    return _start_managed_process(command)


def _environment_value(environment: dict[str, str], name: str) -> str | None:
    """Read a Windows environment variable without assuming case-sensitive keys."""

    wanted = name.casefold()
    for key, value in environment.items():
        if key.casefold() == wanted and value:
            return value
    return None


def find_moonlight_executable(
    environment: dict[str, str] | None = None,
    *,
    platform: str | None = None,
    which: Callable[[str], str | None] | None = None,
    is_file: Callable[[str], bool] | None = None,
) -> str | None:
    """Find Moonlight on ``PATH`` or in its common Windows install locations."""

    lookup = shutil.which if which is None else which
    for command in ("moonlight", "moonlight-qt", "Moonlight.exe"):
        executable = lookup(command)
        if executable:
            return executable

    current_platform = sys.platform if platform is None else platform
    if not current_platform.startswith("win"):
        return None

    source = dict(os.environ if environment is None else environment)
    program_roots = tuple(
        value
        for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)")
        if (value := _environment_value(source, variable))
    )
    local_app_data = _environment_value(source, "LOCALAPPDATA")
    candidates = [
        ntpath.join(root, folder, "Moonlight.exe")
        for root in program_roots
        for folder in ("Moonlight Game Streaming", "Moonlight")
    ]
    if local_app_data:
        candidates.extend(
            (
                ntpath.join(
                    local_app_data,
                    "Programs",
                    "Moonlight Game Streaming",
                    "Moonlight.exe",
                ),
                ntpath.join(
                    local_app_data, "Programs", "Moonlight", "Moonlight.exe"
                ),
                ntpath.join(
                    local_app_data, "Moonlight Game Streaming", "Moonlight.exe"
                ),
            )
        )

    file_exists = os.path.isfile if is_file is None else is_file
    checked: set[str] = set()
    for candidate in candidates:
        normalized = ntpath.normcase(candidate)
        if normalized in checked:
            continue
        checked.add(normalized)
        if file_exists(candidate):
            return candidate
    return None


def external_process_environment(
    environment: dict[str, str] | None = None,
    *,
    frozen: bool | None = None,
    platform: str | None = None,
) -> dict[str, str]:
    """Return an environment safe for external apps launched by PyInstaller.

    A frozen process adjusts native-library and Qt plugin paths for bundled
    libraries. Those libraries may be ABI-incompatible with a system app such
    as Moonlight, so strip bundle-only Qt paths on every platform and restore
    Linux's pre-bundle library search path for external children.
    """

    result = dict(os.environ if environment is None else environment)
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    current_platform = sys.platform if platform is None else platform
    if is_frozen:
        for variable in (
            "QT_PLUGIN_PATH",
            "QML2_IMPORT_PATH",
            "QT_QPA_PLATFORM_PLUGIN_PATH",
        ):
            result.pop(variable, None)
    if is_frozen and current_platform.startswith("linux"):
        original = result.get("LD_LIBRARY_PATH_ORIG")
        if original is None:
            result.pop("LD_LIBRARY_PATH", None)
        else:
            result["LD_LIBRARY_PATH"] = original
    return result


def _set_windows_dll_directory(directory: str | None) -> None:
    """Set the process DLL search directory and surface a Windows API failure."""

    import ctypes

    setter = ctypes.windll.kernel32.SetDllDirectoryW  # type: ignore[attr-defined]
    setter.argtypes = [ctypes.c_wchar_p]
    setter.restype = ctypes.c_int
    if not setter(directory):
        raise ctypes.WinError()


def start_external_process(
    command: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
    frozen: bool | None = None,
    platform: str | None = None,
    bundle_directory: str | os.PathLike[str] | None = None,
    popen_factory: Callable[..., subprocess.Popen[bytes]] | None = None,
) -> subprocess.Popen[bytes]:
    """Start a detached external app without leaking PyInstaller DLL state.

    PyInstaller calls ``SetDllDirectoryW(_MEIPASS)`` on Windows. That setting is
    inherited by child processes and can make an installed Moonlight load the
    bundled application's DLLs. Clear it only around ``CreateProcess``, then
    restore the bundle directory in ``finally`` so this application's imports
    continue using their normal DLL search state even when spawning fails.
    """

    current_platform = sys.platform if platform is None else platform
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    options: dict[str, object] = {}
    if current_platform.startswith("win"):
        options["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        options["start_new_session"] = True

    factory = subprocess.Popen if popen_factory is None else popen_factory

    def spawn() -> subprocess.Popen[bytes]:
        return factory(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=external_process_environment(
                environment,
                frozen=is_frozen,
                platform=current_platform,
            ),
            **options,
        )

    if not (is_frozen and current_platform.startswith("win")):
        return spawn()

    restore_directory = bundle_directory
    if restore_directory is None:
        restore_directory = getattr(sys, "_MEIPASS", None)
    if restore_directory is None:
        # ``_MEIPASS`` is guaranteed by PyInstaller. This fallback keeps a
        # custom frozen bootloader safe rather than clearing state permanently.
        restore_directory = os.path.dirname(sys.executable)

    _set_windows_dll_directory(None)
    try:
        return spawn()
    finally:
        _set_windows_dll_directory(os.fspath(restore_directory))


def request_stop(process: subprocess.Popen[str]) -> None:
    """Ask a whole managed process group to stop cleanly."""

    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
        else:
            os.killpg(process.pid, signal.SIGINT)
    except (OSError, ProcessLookupError, ValueError):
        try:
            process.terminate()
        except OSError:
            pass


def force_stop(process: subprocess.Popen[str]) -> None:
    """Terminate a managed group that ignored the polite request."""

    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            pass


def kill_process(process: subprocess.Popen[str]) -> None:
    """Kill a managed group as a last resort."""

    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass
