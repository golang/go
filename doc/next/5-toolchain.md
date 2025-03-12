## Compiler {#compiler}

<!-- https://go.dev/issue/26379 -->

The compiler and linker in Go 1.25 now generate debug information
using [DWARF version 5](https://dwarfstd.org/dwarf5std.html); the
newer DWARF version reduces the space required for debuging
information in Go binaries.
DWARF 5 generation is gated by the "dwarf5" GOEXPERIMENT; this
functionality can be disabled (for now) using GOEXPERIMENT=nodwarf5.

## Assembler {#assembler}

## Linker {#linker}

