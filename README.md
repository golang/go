# The Go Programming Language

## Shared-toolchain fork

This fork adds an optional shared-library build for Go command-line tools. On
Windows amd64, each tool has a C-compatible DLL entry point and a small native
`.exe` launcher. The launcher loads the matching DLL from its own directory,
forwards arguments and the inherited standard streams, and returns the tool's
exit status. The original native Go executable build remains available.

The current implementation target is Windows amd64. Linux amd64 and macOS
arm64/amd64 are not provided by this build yet. Do not treat those platforms as
supported until their CI builds and smoke tests are present and passing.

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

### Layout and FFI

The Go DLL sits beside its launcher. For example, `bin\go.exe` loads
`bin\go.dll`, and `pkg\tool\windows_amd64\compile.exe` loads the corresponding
`compile.dll`. This keeps Go's regular tool discovery and child-process calls
in place. Libraries export `dll_<tool>_main(int argc, char **argv)` and receive
UTF-8 arguments. A C host can load a DLL and call that symbol directly; set
`GOROOT` in the host process when the host executable is outside the Go root.

The Go runtime is process-wide. The launcher leaves the DLL loaded until
process exit. The direct FFI smoke test calls `go` twice in one process; other
tools should be independently validated before relying on repeated calls in a
long-lived host. Commands retain Go's normal process behavior for operations
that still call `os.Exit` outside the shared-entry exit path.

### CI artifacts

The `Shared toolchain (Windows amd64)` GitHub Actions workflow bootstraps Go,
builds the shared tools and launchers, and exercises CLI, compiler-subprocess,
standard-library, cross-compilation, and direct-FFI paths. A successful run
publishes a downloadable `go-shared-windows-amd64` archive. Linux and macOS
artifacts will be added after platform-specific build and runtime validation.

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
