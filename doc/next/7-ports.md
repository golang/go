## Ports {#ports}

### Darwin {#darwin}

<!-- go.dev/issue/64207 -->
As [announced](go1.22#darwin) in the Go 1.22 release notes,
Go 1.23 requires macOS 11 Big Sur or later;
support for previous versions has been discontinued.

### Linux {#linux}

<!-- go.dev/issue/67001 -->
Go 1.23 is the last release that requires Linux kernel version 2.6.32 or later. Go 1.24 will require Linux kernel version 3.17 or later, with an exception that systems running 3.10 or later will continue to be supported if the kernel has been patched to support the getrandom system call.

### OpenBSD {#openbsd}

<!-- go.dev/issue/55999, CL 518629, CL 518630 -->
Go 1.23 adds experimental support for OpenBSD on 64-bit RISC-V (`GOOS=openbsd`, `GOARCH=riscv64`).

### ARM64 {#arm64}

<!-- go.dev/issue/60905, CL 559555 -->
Go 1.23 introduces a new `GOARM64` environment variable, which specifies the minimum target version of the ARM64 architecture at compile time. Allowed values are `v8.{0-9}` and `v9.{0-5}`. This may be followed by an option specifying extensions implemented by target hardware. Valid options are `,lse` and `,crypto`.

The `GOARM64` environment variable defaults to `v8.0`.

### Wasm {#wasm}

<!-- go.dev/issue/63718 -->
The `go_wasip1_wasm_exec` script in `GOROOT/misc/wasm` has dropped support
for versions of `wasmtime` < 14.0.0.
