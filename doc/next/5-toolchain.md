## Compiler {#compiler}

The build time overhead to building with [Profile Guided Optimization](/doc/pgo) has been reduced significantly.
Previously, large builds could see 100%+ build time increase from enabling PGO.
In Go 1.23, overhead should be in the single digit percentages.

<!-- https://go.dev/issue/62737 , https://golang.org/cl/576681,  https://golang.org/cl/577615 -->
The compiler in Go 1.23 can now overlap the stack frame slots of local variables
accessed in disjoint regions of a function, which reduces stack usage
for Go applications.

<!-- https://go.dev/cl/577935 -->
For 386 and amd64, the compiler will use information from PGO to align certain
hot blocks in loops.  This improves performance an additional 1-1.5% at
a cost of an additional 0.1% text and binary size.  This is currently only implemented
on 386 and amd64 because it has not shown an improvement on other platforms.
Hot block alignment can be disabled with `-gcflags=[<packages>=]-d=alignhot=0`

## Assembler {#assembler}

## Linker {#linker}

<!-- go.dev/issue/67401, CL 585556, CL 587220, and many more -->
TODO: Say what needs to be said in Go 1.23 release notes regarding
the locking down of future linkname uses.

<!-- CL 473495 -->
When building a dynamically linked ELF binary (including PIE binary), the
new `-bindnow` flag enables immediate function binding.
