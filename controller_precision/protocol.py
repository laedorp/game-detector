"""Private line protocol shared by the controller worker and launcher."""

from __future__ import annotations


# This line is consumed, not displayed, by the launcher.  It is intentionally
# exact and versioned so ordinary human-readable output can never be mistaken
# for proof that evdev and uinput are open.
CONTROLLER_READY_SENTINEL = "GAME_DETECTOR_INTERNAL_CONTROLLER_PRECISION_READY_V1"


def is_controller_ready_line(line: str) -> bool:
    """Return true only for the exact worker-ready protocol line."""

    return line.rstrip("\r\n") == CONTROLLER_READY_SENTINEL
