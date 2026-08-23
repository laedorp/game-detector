# Release dependency lock contract

ProAim source installs intentionally retain readable version ranges in
`requirements.txt`. Published Linux CPU and Windows DirectML bundles, and every
Windows CUDA release candidate, use a stricter target-specific contract:

| profile | target | platform lock |
| --- | --- | --- |
| `linux-cpu-py313` | Linux x86-64, Ubuntu 22.04 glibc baseline | `requirements-locks/linux-cpu-py313.txt` |
| `windows-directml-py313` | Windows x86-64, DirectML | `requirements-locks/windows-directml-py313.txt` |
| `windows-cuda-py313` | Windows x86-64, CUDA 13/cuDNN 9 | `requirements-locks/windows-cuda-py313.txt` |

All three profiles use CPython 3.13.14 and
`requirements-locks/bootstrap-py313.txt`. Every direct, transitive, and build
distribution is an exact `name==version` pin with the SHA-256 of the one
reviewed target wheel or sdist. Pip installs the files with `--require-hashes`.
The Windows CUDA lock explicitly enables ONNX Runtime's `cuda,cudnn` extras and
pins the seven resulting NVIDIA distributions and Windows wheels.

The Linux profile pins PySide6 Essentials and shiboken6 6.9.3 because the 6.11
Linux wheels require a newer glibc than the Ubuntu 22.04 bundle baseline. The
Windows profiles can use 6.11.1. Linux `evdev` has no CPython 3.13 manylinux
wheel, so the lock permits exactly the 1.9.3 sdist; build isolation is disabled
and the separately hash-locked setuptools/wheel toolchain builds it.

## Build transaction and verification

Release workflows create a new `.release-venv`; an existing developer
environment is never reused. They install the bootstrap lock, then
force-reinstall the complete bootstrap plus target lock in one final
`--require-hashes --no-compile --no-build-isolation` transaction. Bytecode
writing is disabled for the release jobs so pip does not create unhashed cache
entries. Both pip JSON reports are retained under ignored
`.release-metadata/` paths.

Before PyInstaller runs, `scripts/write_dependency_manifest.py` verifies:

- exact CPython patch, operating system, and x86-64 architecture;
- the last pip report was produced by the locked pip version and individually
  accounts for the complete final distribution set (the bootstrap report
  cannot be substituted or used to hide an inherited package);
- every target-active declaration in `requirements.txt`,
  `requirements-build.txt`, and the chosen runtime file against the lock,
  including required extras;
- pip-report URLs are plain HTTPS artifacts from `files.pythonhosted.org` and
  each reported artifact SHA-256 is permitted by the repository lock;
- the installed distribution set exactly equals the lock, with no missing,
  extra, duplicate, wrong-version, or stacked ONNX Runtime distribution;
- every installed file listed by each distribution's PEP 376 `RECORD` has the
  recorded SHA-256 and byte size; only the necessarily self-referential
  `RECORD` row may be unhashed, and the verifier records a deterministic
  aggregate of all actual installed file hashes and sizes;
- the union of those `RECORD` paths exactly covers every regular file below the
  installed package roots, with no unowned module, `.pth`, native library,
  bytecode cache, symlink, missing entry, or file claimed by two distributions;
- `pip check` succeeds.

It writes `DEPENDENCY-MANIFEST.json` with the lock/input hashes, exact artifact
hash and filename for every distribution, installed METADATA/RECORD and payload
aggregates, and the actual interpreter hash. The build helper places it beside
the executable; `BUILD-INFO.json` schema 2 binds its SHA-256, profile, and
distribution count. The same BUILD-INFO record also binds the launcher's
canonical release-default ONNX preset, `[height,width]` input shape, safe
bundle-relative model and labels paths, and both file SHA-256 values. This
keeps qualification attached to the model the application actually selects,
even when a future release changes the default preset or tensor shape.
CUDA candidate inspection and physical qualification revalidate that binding.

## Updating a lock

Do not edit a version or hash only to silence pip. A lock update requires all
of the following:

1. Resolve from a new CPython 3.13.14 virtual environment on the target release
   runner. Download the exact target artifacts and independently calculate
   their SHA-256 values. For Windows, use CPython 3.13, ABI `cp313`, platform
   `win_amd64`; for Linux use the Ubuntu 22.04-compatible manylinux artifacts
   and the reviewed `evdev` sdist.
2. Run `pip download --require-hashes` for the bootstrap plus updated platform
   lock. Then perform the real fresh-environment install used by the workflow,
   run the dependency-manifest verifier, and run `pip check`.
3. Review upstream license/security/release notes and the resulting installed
   native libraries. An ONNX Runtime, CUDA, cuDNN, DirectML, OpenVINO, Qt, or
   capture change also requires matching physical-hardware qualification; a
   dependency resolution alone never qualifies hardware.
4. Run the full repository test and frozen-bundle gates on Linux and Windows.
   Retain the pip reports and resulting bundle manifest with the release
   evidence.

## Reproducibility boundary

The locks make Python package selection and package bytes fail closed. They do
not promise byte-identical ZIPs: CPython itself, GitHub runner images, OS
packages, the C compiler used for `evdev`, PyInstaller timestamps/order, GPU
drivers, and signing/archiving tools remain separate inputs. The bundle records
the actual interpreter and installed metadata so those builds are auditable.
Changing any of those inputs still requires a new build and qualification; two
bundles should be compared by their recorded manifests and functional gates,
not assumed identical from the package lock alone.
