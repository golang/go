## Compiler {#compiler}

The build time overhead to building with [Profile Guided Optimization](/doc/pgo) has been reduced significantly.
Previously, large builds could see 100%+ build time increase from enabling PGO.
In Go 1.23, overhead should be in the single digit percentages.

<!-- https://go.dev/issue/62737 , https://golang.org/cl/576681,  https://golang.org/cl/577615 -->
The compiler in Go 1.23 can now overlap the stack frame slots of local variables
accessed in disjoint regions of a function, which reduces stack usage
for Go applications.

## Assembler {#assembler}

## Linker {#linker}


