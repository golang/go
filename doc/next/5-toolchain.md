## Compiler {#compiler}

<!-- go.dev/issue/60725, go.dev/issue/57926 -->
The compiler already disallowed defining new methods with receiver types that were
cgo-generated, but it was possible to circumvent that restriction via an alias type.
Go 1.24 now always reports an error if a receiver denotes a cgo-generated type,
whether directly or indirectly (through an alias type).

## Assembler {#assembler}

## Linker {#linker}

<!-- go.dev/issue/68678, go.dev/issue/68652, CL 618598, CL 618601 -->
The linker now generates a GNU build ID (the ELF `NT_GNU_BUILD_ID` note) on ELF platforms
and a UUID (the Mach-O `LC_UUID` load command) on macOS by default.
The build ID or UUID is derived from the Go build ID.
It can be disabled by the `-B none` linker flag, or overridden by the `-B 0xNNNN` linker
flag with a user-specified hexadecimal value.

## Bootstrap {#bootstrap}

<!-- go.dev/issue/64751 -->
As mentioned in the [Go 1.22 release notes](/doc/go1.22#bootstrap), Go 1.24 now requires
Go 1.22.6 or later for bootstrap.
We expect that Go 1.26 will require a point release of Go 1.24 or later for bootstrap.
