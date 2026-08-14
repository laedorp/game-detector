"""Shared privacy checks for evidence that may leave the workstation."""

from __future__ import annotations

import re


_PUBLIC_WEB_SCHEME = re.compile(r"(?i)\bhttps?://")
_FILE_URI = re.compile(r"(?i)(?<![A-Za-z0-9])file:(?://|[\\/]|[A-Za-z]:[\\/])")
_DRIVE_ABSOLUTE = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_HOME_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9._~-])~[\\/]")
_ROOTED_PATH = re.compile(r"(?<![A-Za-z0-9._~/-])[\\/]")


def contains_nonportable_path(value: str) -> bool:
    """Return whether *value* embeds a local absolute path or ``file:`` URI.

    Detection uses path syntax rather than a finite list of prose or CSV
    delimiters. Ordinary relative artifact names and public HTTP(S) URL paths
    remain valid. A local path hidden in a URL query or fragment is rejected.
    """

    if not isinstance(value, str):
        raise TypeError("public evidence path scanning requires a string")

    # Remove only the public scheme marker. Normal URL path separators remain
    # preceded by host/path characters, while ``?loaded_from=/tmp/model``
    # still looks like a rooted local path.
    protected = _PUBLIC_WEB_SCHEME.sub("https__", value)
    return bool(
        "\\" in protected
        or _FILE_URI.search(protected)
        or _DRIVE_ABSOLUTE.search(protected)
        or _HOME_ABSOLUTE.search(protected)
        or _ROOTED_PATH.search(protected)
    )
