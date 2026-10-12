"""Python ctypes launcher for a sibling Go c-shared tool DLL."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys


def main() -> int:
    script = Path(__file__).resolve()
    tool = script.stem
    library_path = script.with_suffix(".dll")
    executable_path = script.with_suffix(".exe")
    # Keep the native tool's argv[0] spelling even though the launcher itself
    # is a .py file. Several Go tools derive their displayed name from argv[0]
    # and `go` parses that output (for example `compile -V=full`).
    argv0 = str(executable_path)

    # Find this distribution's root before loading the Go runtime. Preserve
    # an explicit GOROOT only when it looks like a Go source/distribution root;
    # otherwise a stale setting (for example pkg/tool/windows_amd64) makes
    # cmd/go search for the standard library under the tool directory.
    distribution_root = next(
        (
            parent
            for parent in script.parents
            if (parent / "src" / "cmd" / "go").is_dir()
            and (parent / "pkg" / "tool").is_dir()
        ),
        None,
    )
    configured_root = os.environ.get("GOROOT")
    if distribution_root is not None and (
        not configured_root
        or not (Path(configured_root) / "src" / "cmd" / "go").is_dir()
    ):
        os.environ["GOROOT"] = str(distribution_root)

    try:
        # Include this directory when Windows resolves the DLL's dependencies.
        dll_directory = os.add_dll_directory(str(script.parent)) if hasattr(os, "add_dll_directory") else None
        library = ctypes.CDLL(str(library_path))
    except OSError as error:
        print(f"python launcher: cannot load {library_path}: {error}", file=sys.stderr)
        return 127

    try:
        entry = getattr(library, f"dll_{tool}_main")
    except AttributeError:
        print(f"python launcher: missing entry point dll_{tool}_main in {library_path}", file=sys.stderr)
        return 127

    entry.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
    entry.restype = ctypes.c_int
    arguments = [argv0, *sys.argv[1:]]
    encoded = [argument.encode("utf-8", "surrogateescape") for argument in arguments]
    argv = (ctypes.c_char_p * len(encoded))(*encoded)

    # Keep the DLL search directory registered through the FFI call.
    try:
        return int(entry(len(encoded), argv))
    finally:
        if dll_directory is not None:
            dll_directory.close()


if __name__ == "__main__":
    raise SystemExit(main())
