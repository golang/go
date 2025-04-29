## Compiler {#compiler}

<!-- https://go.dev/issue/26379 -->

The compiler and linker in Go 1.25 now generate debug information
using [DWARF version 5](https://dwarfstd.org/dwarf5std.html); the
newer DWARF version reduces the space required for debugging
information in Go binaries.
DWARF 5 generation is gated by the "dwarf5" GOEXPERIMENT; this
functionality can be disabled (for now) using GOEXPERIMENT=nodwarf5.

<!-- https://go.dev/issue/72860, CL 657715 -->

The compiler [has been fixed](/cl/657715)
to ensure that nil pointer checks are performed promptly. Programs like the following,
which used to execute successfully, will now panic with a nil-pointer exception:

```
package main

import "os"

func main() {
	f, err := os.Open("nonExistentFile")
	name := f.Name()
	if err != nil {
		return
	}
	println(name)
}
```

This program is incorrect in that it uses the result of `os.Open` before checking
the error. The main result of `os.Open` can be a nil pointer if the error result is non-nil.
But because of [a compiler bug](/issue/72860), this program ran successfully under
Go versions 1.21 through 1.24 (in violation of the Go spec). It will no longer run
successfully in Go 1.25. If this change is affecting your code, the solution is to put
the non-nil error check earlier in your code, preferably immediately after
the error-generating statement.

## Assembler {#assembler}

## Linker {#linker}

