from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


class ExportArtifactError(RuntimeError):
    """Raised when an OpenVINO export does not contain one safe IR pair."""


def _artifact_basename(value: str) -> str:
    basename = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", basename):
        raise argparse.ArgumentTypeError(
            "must be a 1-80 character filename stem using letters, digits, _ or -"
        )
    return basename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download/export a YOLO detect model to an OpenVINO IR directory."
    )
    parser.add_argument("--weights", default="yolo26n.pt", help="Ultralytics weights or local .pt file.")
    parser.add_argument("--imgsz", type=int, default=320, help="Square export size (default: 320).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/yolo26n_openvino_model"),
        help="Destination directory for the exported IR.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export dynamic spatial dimensions; static input is faster to validate and is the default.",
    )
    parser.add_argument(
        "--basename",
        type=_artifact_basename,
        help=(
            "Final filename stem for the OpenVINO .xml/.bin pair. By default, "
            "keep the stem generated from --weights."
        ),
    )
    return parser


def _find_ir_pair(directory: Path) -> tuple[Path, Path]:
    try:
        children = list(directory.iterdir())
    except OSError as exc:
        raise ExportArtifactError(
            f"Cannot inspect exported OpenVINO directory {directory}: {exc}"
        ) from exc

    xml_files = sorted(
        path
        for path in children
        if path.suffix == ".xml" and path.is_file() and not path.is_symlink()
    )
    bin_files = sorted(
        path
        for path in children
        if path.suffix == ".bin" and path.is_file() and not path.is_symlink()
    )
    if len(xml_files) != 1 or len(bin_files) != 1:
        raise ExportArtifactError(
            "Expected exactly one regular .xml file and one regular .bin file in "
            f"the OpenVINO export, found {len(xml_files)} and {len(bin_files)}: {directory}"
        )
    xml_file, bin_file = xml_files[0], bin_files[0]
    if xml_file.stem != bin_file.stem:
        raise ExportArtifactError(
            "OpenVINO .xml and .bin filenames do not have the same stem: "
            f"{xml_file.name}, {bin_file.name}"
        )
    return xml_file, bin_file


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def normalize_ir_basename(directory: Path, basename: str) -> tuple[Path, Path]:
    """Give one verified IR pair a stable stem without replacing existing files.

    Hard links make each target creation an atomic no-replace operation. The old
    names remain in place until both new names exist, so a failed second link can
    be rolled back without leaving a half-renamed model pair.
    """

    safe_basename = _artifact_basename(basename)
    source_xml, source_bin = _find_ir_pair(directory)
    target_xml = directory / f"{safe_basename}.xml"
    target_bin = directory / f"{safe_basename}.bin"
    pairs = ((source_xml, target_xml), (source_bin, target_bin))
    if all(source == target for source, target in pairs):
        return target_xml, target_bin

    collisions = [
        target.name
        for source, target in pairs
        if source != target and _path_lexists(target)
    ]
    if collisions:
        raise ExportArtifactError(
            "Refusing to overwrite existing OpenVINO artifact(s): "
            + ", ".join(collisions)
        )

    created: list[Path] = []
    try:
        for source, target in pairs:
            os.link(source, target)
            created.append(target)
    except OSError as exc:
        for target in reversed(created):
            try:
                target.unlink()
            except OSError:
                pass
        raise ExportArtifactError(
            f"Could not safely rename the OpenVINO artifact pair: {exc}"
        ) from exc

    try:
        source_xml.unlink()
        source_bin.unlink()
    except OSError as exc:
        # Restore any old name already removed, then remove the new links. This
        # keeps the exporter-generated pair usable if final cleanup fails.
        for source, target in pairs:
            if not _path_lexists(source):
                try:
                    os.link(target, source)
                except OSError:
                    pass
        if all(_path_lexists(source) for source, _ in pairs):
            for target in reversed(created):
                try:
                    target.unlink()
                except OSError:
                    pass
        raise ExportArtifactError(
            f"Could not remove the exporter-generated artifact names: {exc}"
        ) from exc

    return target_xml, target_bin


def main() -> None:
    args = build_parser().parse_args()
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be greater than zero")

    output = args.output.expanduser().resolve()
    if _path_lexists(output):
        raise SystemExit(f"Output already exists: {output}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required only for export. Install requirements-export.txt "
            "in a separate export environment."
        ) from exc

    weights = Path(args.weights).expanduser()
    weights_arg = str(weights.resolve()) if weights.exists() else args.weights
    model = YOLO(weights_arg)
    if getattr(model, "task", None) != "detect":
        raise SystemExit(
            f"Unsupported model task {getattr(model, 'task', None)!r}; "
            "this runtime accepts detection models only."
        )
    exported = Path(
        model.export(
            format="openvino",
            imgsz=args.imgsz,
            batch=1,
            dynamic=args.dynamic,
            device="cpu",
        )
    ).resolve()
    exported_dir = exported if exported.is_dir() else exported.parent

    try:
        if args.basename is None:
            exported_xml, exported_bin = _find_ir_pair(exported_dir)
        else:
            exported_xml, exported_bin = normalize_ir_basename(exported_dir, args.basename)
    except ExportArtifactError as exc:
        raise SystemExit(f"Invalid OpenVINO export: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    if exported_dir == output:
        print(
            f"OpenVINO model exported to {output} "
            f"({exported_xml.name}, {exported_bin.name})"
        )
        return

    if _path_lexists(output):
        raise SystemExit(f"Output appeared during export; refusing to overwrite it: {output}")
    shutil.move(str(exported_dir), str(output))
    print(
        f"OpenVINO model exported to {output} "
        f"({exported_xml.name}, {exported_bin.name})"
    )


if __name__ == "__main__":
    main()
