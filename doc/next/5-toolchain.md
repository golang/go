## Compiler {#compiler}

<!-- go.dev/issue/60725, go.dev/issue/57926 -->
The compiler already disallowed defining new methods with receiver types that were
cgo-generated, but it was possible to circumvent that restriction via an alias type.
Go 1.24 now always reports an error if a receiver denotes a cgo-generated type,
whether directly or indirectly (through an alias type).

## Assembler {#assembler}

## Linker {#linker}

## Bootstrap {#bootstrap}

<!-- go.dev/issue/64751 -->
As mentioned in the [Go 1.22 release notes](/doc/go1.22#bootstrap), Go 1.24 now requires
Go 1.22.6 or later for bootstrap.
We expect that Go 1.26 will require a point release of Go 1.24 or later for bootstrap.
