## Compiler {#compiler}

## Assembler {#assembler}

## Linker {#linker}

On 64-bit ARM-based Windows (the `windows/arm64` port), the linker now supports internal
linking mode of cgo programs, which can be requested with the
`-ldflags=-linkmode=internal` flag.

## Bootstrap {#bootstrap}

<!-- go.dev/issue/69315 -->
As mentioned in the [Go 1.24 release notes](/doc/go1.24#bootstrap), Go 1.26 now requires
Go 1.24.6 or later for bootstrap.
We expect that Go 1.28 will require a minor release of Go 1.26 or later for bootstrap.
