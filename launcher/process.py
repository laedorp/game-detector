"""Child-process lifecycle helpers used by the Tk launcher."""

from __future__ import annotations

import os
import signal
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


def external_process_environment(
    environment: dict[str, str] | None = None,
    *,
    frozen: bool | None = None,
    platform: str | None = None,
) -> dict[str, str]:
    """Return an environment safe for external apps launched by PyInstaller.

    A frozen Linux process adjusts its library search path for bundled native
    libraries.  Those libraries may be ABI-incompatible with a system app such
    as Moonlight, so restore the pre-bundle value for external children.
    """

    result = dict(os.environ if environment is None else environment)
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    current_platform = sys.platform if platform is None else platform
    if is_frozen and current_platform.startswith("linux"):
        original = result.get("LD_LIBRARY_PATH_ORIG")
        if original is None:
            result.pop("LD_LIBRARY_PATH", None)
        else:
            result["LD_LIBRARY_PATH"] = original
    return result


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
