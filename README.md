# The Go Programming Language

## Shared-toolchain fork

This fork adds optional shared-library builds for Go command-line tools. On
Windows amd64, each tool has a C-compatible DLL entry point and a small native
`.exe` launcher. Linux amd64 uses `.so` libraries and native launchers. Each
launcher loads its matching library from its own directory, forwards arguments
and inherited standard streams, and returns the tool's exit status. The
original native Go executable build remains available.

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
root.

The Go runtime is process-wide. The launcher leaves the DLL loaded until
process exit. The direct FFI smoke test calls `go` twice in one process; other
tools should be independently validated before relying on repeated calls in a
long-lived host. Commands retain Go's normal process behavior for operations
that still call `os.Exit` outside the shared-entry exit path.

### CI artifacts

The `Shared toolchain (Windows amd64)` and `Shared toolchain (Linux amd64)`
GitHub Actions workflows bootstrap Go, build the shared tools and launchers,
and exercise CLI, compiler-subprocess, standard-library, cross-compilation,
and direct-FFI paths. Successful runs publish downloadable Windows ZIP and
Linux tar.gz artifacts. macOS arm64/amd64 remains unimplemented and untested.

The independent `Shared toolchain release (Windows + Linux amd64)` workflow
integrates both platform builds, validation, packaging, and GitHub Release
publishing. Push a tag matching `shared-toolchain-v*` to run the complete
pipeline. For example, use `shared-toolchain-v0.1.0` for the first release.
Only after both platform jobs pass does the workflow create or update the
release and attach the Windows ZIP, Linux tar.gz, and `SHA256SUMS`. The same
run also publishes both packages as downloadable CI artifacts.

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
