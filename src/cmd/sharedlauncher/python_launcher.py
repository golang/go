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
    argv0 = str(executable_path if executable_path.is_file() else script)

    # Set GOROOT before loading the Go runtime when the distribution root can
    # be identified from this script's location.
    for parent in script.parents:
        if (parent / "src" / "cmd" / "go").is_dir() and (parent / "pkg" / "tool").is_dir():
            os.environ.setdefault("GOROOT", str(parent))
            break

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
