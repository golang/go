## Ports {#ports}

### Darwin {#darwin}

<!-- go.dev/issue/64207 -->
As [announced](go1.22#darwin) in the Go 1.22 release notes,
Go 1.23 requires macOS 11 Big Sur or later;
support for previous versions has been discontinued.

### Wasm {#wasm}

<!-- go.dev/issue/63718 -->
The `go_wasip1_wasm_exec` script in `GOROOT/misc/wasm` has dropped support
for versions of `wasmtime` < 14.0.0.
