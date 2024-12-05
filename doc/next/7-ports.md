## Ports {#ports}

### Linux {#linux}

<!-- go.dev/issue/67001 -->
As [announced](go1.23#linux) in the Go 1.23 release notes, Go 1.24 requires Linux
kernel version 3.2 or later.

### Darwin {#darwin}

<!-- go.dev/issue/69839 -->
Go 1.24 is the last release that will run on macOS 11 Big Sur.
Go 1.25 will require macOS 12 Monterey or later.

### WebAssembly {#wasm}

<!-- go.dev/issue/65199, CL 603055 -->
The `go:wasmexport` directive is added for Go programs to export functions to the WebAssembly host.

On WebAssembly System Interface Preview 1 (`GOOS=wasip1, GOARCH=wasm`), Go 1.24 supports
building a Go program as a
[reactor/library](https://github.com/WebAssembly/WASI/blob/63a46f61052a21bfab75a76558485cf097c0dbba/legacy/application-abi.md#current-unstable-abi),
by specifying the `-buildmode=c-shared` build flag.

<!-- go.dev/issue/66984, CL 626615 -->
More types are now permitted as argument or result types for `go:wasmimport` functions.
Specifically, `bool`, `string`, `uintptr`, and pointers to certain types are allowed
(see the [proposal](/issue/66984) for detail),
along with 32-bit and 64-bit integer and float types, and `unsafe.Pointer`, which
are already allowed.
These types are also permitted as argument or result types for `go:wasmexport` functions.

<!-- go.dev/issue/68024 -->
The support files for WebAssembly have been moved to `lib/wasm` from `misc/wasm`.
