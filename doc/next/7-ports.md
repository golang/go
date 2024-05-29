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
<!-- TODO: Should it say 'experimental' like in go.dev/doc/go1.22#openbsd or https://go.dev/doc/go1.20#freebsd-riscv, or not? -->
Go 1.23 adds experimental support for OpenBSD on RISC-V (`GOOS=openbsd`, `GOARCH=riscv64`).

### Wasm {#wasm}

<!-- go.dev/issue/63718 -->
The `go_wasip1_wasm_exec` script in `GOROOT/misc/wasm` has dropped support
for versions of `wasmtime` < 14.0.0.
