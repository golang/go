# The Go Programming Language

## Shared-toolchain fork

This fork adds optional shared-library builds for Go command-line tools. On
Windows amd64, each tool has a C-compatible DLL entry point, a small native
`.exe` launcher, a Python `ctypes` launcher, and a `.bat` wrapper for the Python
launcher. Linux amd64 uses `.so` libraries and native launchers. The launchers
load their matching library from their own directory, forward arguments and
inherited standard streams, and return the tool's exit status. The original
native Go executable build remains available.

### Build Windows amd64 from source

Install the Go 1.26 bootstrap toolchain and MinGW-w64 GCC. From the repository
root, run:

```bat
cd src
make.bat
cd ..
src\make-shared.bat
```

`make-shared.bat` builds shared versions of `go`, `gofmt`, `asm`, `cgo`,
`compile`, `link`, `preprofile`, `cover`, `export`, `fix`, and `vet`. The native
Go build must succeed before that step. The script saves the original native
executables under `native\`; run `src\use-native.bat` to restore them.
The generated `bin\go.py`, `bin\go.bat`, and corresponding files under
`pkg\tool\windows_amd64` call the sibling DLLs through Python's standard
`ctypes` module. These optional launchers require Python 3 on `PATH` (or the
Windows `py` launcher); the native `.exe` launchers do not require Python.

### Build Linux amd64 from source

Install the Go 1.26 bootstrap toolchain and a C compiler. From the repository
root, run:

```sh
cd src && ./make.bash
cd ..
sh src/make-shared.sh
```

The script saves the native executables under `native/`; run
`sh src/use-native.sh` to restore them.

### Layout and FFI

The shared library sits beside its launcher. For example, `bin\go.exe` loads
`bin\go.dll`, `bin/go` loads `bin/go.so`, and each platform's
`pkg/tool/<host_os>_<host_arch>` directory contains matching tool launchers and
libraries. This keeps Go's regular tool discovery and child-process calls in
place. Libraries export `dll_<tool>_main(int argc, char **argv)` and receive
UTF-8 arguments. A C host can load the library and call that symbol directly;
set `GOROOT` in the host process when the host executable is outside the Go
root. On Windows, the generated Python scripts load the same DLL with
`ctypes.CDLL` and invoke that entry point directly.

The Go runtime is process-wide. The launcher leaves the DLL loaded until
process exit. The direct FFI smoke test calls `go` twice in one process; other
tools should be independently validated before relying on repeated calls in a
long-lived host. Commands retain Go's normal process behavior for operations
that still call `os.Exit` outside the shared-entry exit path.

### Releases

The `Shared toolchain release (Windows + Linux amd64)` GitHub Actions workflow
builds and validates both platforms, then attaches the Windows ZIP, Linux
tar.gz, and `SHA256SUMS` to a GitHub Release. Run it manually from the Actions
tab and select a release tag matching `shared-toolchain-v*` in the “Use
workflow from” selector. The workflow also publishes both packages as
downloadable CI artifacts. macOS arm64/amd64 remains unimplemented and
untested.

The reusable native launcher and its Linux smoke test are in
`src/cmd/sharedlauncher/`.

Go is an open source programming language that makes it easy to build simple,
reliable, and efficient software.

![Gopher image](https://golang.org/doc/gopher/fiveyears.jpg)
*Gopher image by [Renee French][rf], licensed under [Creative Commons 4.0 Attribution license][cc4-by].*

Our canonical Git repository is located at https://go.googlesource.com/go.
There is a mirror of the repository at https://github.com/golang/go.

Unless otherwise noted, the Go source files are distributed under the
BSD-style license found in the LICENSE file.

### Download and Install

#### Binary Distributions

Official binary distributions are available at https://go.dev/dl/.

After downloading a binary release, visit https://go.dev/doc/install
for installation instructions.

#### Install From Source

If a binary distribution is not available for your combination of
operating system and architecture, visit
https://go.dev/doc/install/source
for source installation instructions.

### Contributing

Go is the work of thousands of contributors. We appreciate your help!

To contribute, please read the contribution guidelines at https://go.dev/doc/contribute.

Note that the Go project uses the issue tracker for bug reports and
proposals only. See https://go.dev/wiki/Questions for a list of
places to ask questions about the Go language.

[rf]: https://reneefrench.blogspot.com/
[cc4-by]: https://creativecommons.org/licenses/by/4.0/
