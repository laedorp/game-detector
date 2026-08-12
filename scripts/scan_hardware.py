"""Report this machine's compute hardware and the detector setup that fits it.

Run this on any machine before configuring the detector:

    python scripts/scan_hardware.py
    python scripts/scan_hardware.py --json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.hardware import describe, scan_and_recommend  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable output instead of the readable report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile, plans = scan_and_recommend()

    if args.json:
        payload = {
            "system": profile.system,
            "processor": asdict(profile.processor) | {
                "flags": sorted(profile.processor.flags)
            },
            "accelerators": [asdict(item) for item in profile.accelerators],
            "runtime_devices": list(profile.runtime_devices),
            "recommendations": [asdict(plan) for plan in plans],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(describe(profile, plans))
    if not any(plan.ready for plan in plans):
        print("\nNo usable inference device was found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
