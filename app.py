"""Desktop and command-line entry point for Game Detector."""

from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--controller-precision":
        # Keep the Linux controller worker inside the same source/frozen app
        # while leaving it entirely separate from the detector process.
        from controller_precision.cli import main as precision_main

        return precision_main(arguments[1:])
    if arguments and arguments[0] == "--cli":
        # The existing CLI parser reads sys.argv.  Keep that mature code path
        # unchanged while letting the desktop launcher use the same executable.
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0], *arguments[1:]]
            from main import main as detector_main

            return detector_main()
        finally:
            sys.argv = original_argv

    if not arguments or arguments in (["--gui"], ["--qt"]):
        # The Qt interface is the default face of the application.  --tk stays
        # available because the Tk launcher still carries panels the Qt one
        # does not.
        try:
            from launcher.qt_app import run_gui as run_qt_gui
        except ImportError as exc:
            print(
                "The desktop interface needs PySide6. Install it with "
                f"'python -m pip install PySide6-Essentials'.\nDetails: {exc}",
                file=sys.stderr,
            )
            return 3
        return run_qt_gui()

    if arguments != ["--tk"]:
        print(
            "Usage: app.py [--gui] | app.py --tk | app.py --cli [detector options] | "
            "app.py --controller-precision [options]",
            file=sys.stderr,
        )
        return 2

    try:
        from launcher.application import run_gui
    except (ImportError, OSError) as exc:
        print(
            "The desktop interface could not start because Tk is unavailable. "
            "Install your system's Tk/Tkinter package, then try again.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 2
    try:
        return run_gui()
    except Exception as exc:
        # Tk initialization can fail before a message box exists (for example,
        # from a headless shell), so keep the entry point diagnostically useful.
        print(f"The desktop interface could not start: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # PyInstaller workers may re-enter the executable with private
    # multiprocessing flags.  Consume those before our GUI/CLI dispatcher sees
    # them, especially when OpenVINO initializes helper processes.
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())
