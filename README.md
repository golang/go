## Go-Panikint

### Overview

`Go-Panikint` is a modified version of the Go compiler that adds **automatic overflow/underflow detection** for integer arithmetic operations and **type truncation detection** for integer conversions. When overflow or truncation is detected, a **panic** with a detailed error message is triggered, including the specific operation type and integer types involved.

**Arithmetic operations**: Handles addition `+`, subtraction `-`, multiplication `*`, and division `/` for both signed and unsigned integer types. For signed integers, covers `int8`, `int16`, `int32`. For unsigned integers, covers `uint8`, `uint16`, `uint32`, `uint64`. The division case specifically detects the `MIN_INT / -1` overflow condition for signed integers. `int64` and `uintptr` are not checked for arithmetic operations.

**Type truncation detection**: Detects when integer type conversions would result in data loss due to the target type having a smaller range than the source type. Covers all integer types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`. Excludes `uintptr` due to platform-dependent usage.

### Usage and installation :
```bash
# Clone, change dir and compile the compiler
git clone https://github.com/kevin-valerio/go-panikint && cd go-panikint/src && ./make.bash

# Full path to the root of the forked compiler
export GOROOT=/path/to/go-panikint

# Compile and run a Go program
./bin/go run test_simple_overflow.go

# Compile only
./bin/go build test_simple_overflow.go

# Fuzz only
./bin/go test -fuzz=FuzzIntegerOverflow -v
```

### How does it work ?
#### What is being done exactly ?
We basically patched the intermediate representation (IR) part of the Go compiler so that, on every math operands (i.e `OADD`, `OMUL`, `OSUB`, `ODIV`, ...) and type conversions, the compiler does not only add the IR opcodes that perform the operation but also **insert** a bunch of checks for arithmetic bugs and insert panic calls with detailed error messages. Panic messages include specific operation types (e.g., "integer overflow in int8 addition operation", "integer truncation: uint16 cannot fit in uint8"). This code will ultimately end up to the binary code (Assembly) of the application, so use with caution.
Below is an example of a Ghidra-decompiled addition `+`:

```c++
if (*x_00 == '+') {
  val = (uint32)*(undefined8 *)(puVar9 + 0x60);
  sVar23 = val + sVar21;
  puVar17 = puVar9 + 8;
  if (((sdword)val < 0 && sVar21 < 0) && (sdword)val < sVar23 ||
      ((sdword)val >= 0 && sVar21 >= 0) && sVar23 < (sdword)val) {
    runtime.panicoverflow(); // <-- panic if overflow caught
  }
  goto LAB_1000a10d4;
}
```

#### Why do we use source-location-based filtering ?
As implemented in `src/cmd/compile/internal/ssagen/ssa.go`, we apply a source-location-based filtering for overflow detection. This ensures overflow detection is applied only to user code and target applications (like security audits of external codebases) while excluding standard library and third-party dependencies.
Each arithmetic operation (`intAdd`, `intSub`, `intMul`, `intDiv`) checks the actual source file location using `n.Pos()` and `base.Ctxt.PosTable.Pos(pos).Filename()`. Operations from files containing `/go-panikint/src/`, `/pkg/mod/`, `/vendor/` are automatically excluded  and standard library packages (`runtime`, `sync`, `os`, `syscall`, etc.) / internal packages (`internal/*`) are excluded during compiler build.

### Testing

You can run theÒ test suite in `tests/` with:

```bash
cd tests/;
GOROOT=/path/to/go-panikint /path/to/go-panikint/bin/go test -v .
```

### Examples

#### Example 1 (Signed Integer Overflow):

```go
package main

import "fmt"

func main() {
	fmt.Println("Testing signed integer overflow detection...")

	// Test int8 addition overflow
	var a int8 = 127
	var b int8 = 1
	fmt.Printf("Before: a=%d, b=%d\n", a, b)
	result := a + b  // Should panic with "integer overflow in int8 addition operation"
	fmt.Printf("After: result=%d\n", result)
}
```

#### Example 2 (Unsigned Integer Overflow):

```go
package main

import "fmt"

func main() {
	fmt.Println("Testing unsigned integer overflow detection...")

	// Test uint8 addition overflow
	var a uint8 = 255
	var b uint8 = 1
	fmt.Printf("Before: a=%d, b=%d\n", a, b)
	result := a + b  // Should panic with "integer overflow in uint8 addition operation"
	fmt.Printf("After: result=%d\n", result)
}
```

**Expected output (for both signed and unsigned overflow):**

```bash
bash-5.2$ GOROOT=/path/to/go-panikint && ./bin/go run test_overflow.go
Testing overflow detection...
Before: a=127, b=1
panic: runtime error: integer overflow in int8 addition operation

goroutine 1 [running]:
main.main()
	/path/to/go-panikint/test_overflow.go:12 +0xfc
exit status 2
```

#### Example 3 (Type truncation):

```go
package main

import "fmt"

func main() {
	fmt.Println("Testing type truncation detection...")

	var u16 uint16 = 256
	fmt.Printf("Before: u16=%d\n", u16)
	result := uint8(u16)  // Should panic with "integer truncation: uint16 cannot fit in uint8"
	fmt.Printf("After: result=%d\n", result)
}
```

**Expected output:**

```bash
bash-5.2$ GOROOT=/path/to/go-panikint && ./bin/go run test_truncation.go
Testing type truncation detection...
Before: u16=256
panic: runtime error: integer truncation: uint16 cannot fit in uint8

goroutine 1 [running]:
main.main()
	/path/to/go-panikint/test_truncation.go:10 +0xfc
exit status 2
```

#### Example 4 (fuzzing):
**Fuzzing harness:**
```go
package fuzztest
import "testing"

func FuzzIntegerOverflow(f *testing.F) {
	f.Fuzz(func(t *testing.T, a, b int8) {
		result := a + b
		t.Logf("%d + %d = %d", a, b, result)
	})
}
```

**Output:**
```bash
GOROOT=/path/to/go-panikint/go-panikint  ../bin/go test -fuzz=FuzzIntegerOverflow -v
=== RUN   FuzzIntegerOverflow
fuzz: elapsed: 0s, gathering baseline coverage: 0/15 completed
fuzz: elapsed: 0s, gathering baseline coverage: 9/15 completed
--- FAIL: FuzzIntegerOverflow (0.03s)
    --- FAIL: FuzzIntegerOverflow (0.00s)
        testing.go:1822: panic: runtime error: integer overflow in int8 addition operation
            goroutine 23 [running]:
            runtime/debug.Stack()
            	/path/to/go-panikintgo-panikint/src/runtime/debug/stack.go:26 +0xc4
            testing.tRunner.func1()
            	/path/to/go-panikintgo-panikint/src/testing/testing.go:1822 +0x220
            panic({0x100e27b00?, 0x100fa4d60?})
            	/path/to/go-panikintgo-panikint/src/runtime/panic.go:783 +0x120
            fuzztest.FuzzIntegerOverflow.func1(0x0?, 0x0?, 0x0?)
            	/path/to/go-panikintgo-panikint/fuzz_test/fuzz_test.go:10 +0xf8
            reflect.Value.call({0x100e24ca0?, 0x100e5d5f8?, 0x14000093e28?}, {0x100dba042, 0x4}, {0x1400012a180, 0x3, 0x0?})
            	/path/to/go-panikintgo-panikint/src/reflect/value.go:581 +0x960
            reflect.Value.Call({0x100e24ca0?, 0x100e5d5f8?, 0x14000154000?}, {0x1400012a180?, 0x100e5ce40?, 0x100d2abf3?})
            	/path/to/go-panikintgo-panikint/src/reflect/value.go:365 +0x94
            testing.(*F).Fuzz.func1.1(0x140001028c0?)
            	/path/to/go-panikintgo-panikint/src/testing/fuzz.go:341 +0x258
            testing.tRunner(0x140001028c0, 0x14000152000)
            	/path/to/go-panikintgo-panikint/src/testing/testing.go:1931 +0xc8
            created by testing.(*F).Fuzz.func1 in goroutine 7
            	/path/to/go-panikintgo-panikint/src/testing/fuzz.go:328 +0x4a4


    Failing input written to testdata/fuzz/FuzzIntegerOverflow/8a8466cc4de923f0
    To re-run:
    go test -run=FuzzIntegerOverflow/8a8466cc4de923f0
```
