# go-panikint: Go Compiler with Arithmetic Overflow Detection

## Overview

`go-panikint` is a modified version of the Go compiler that adds **automatic overflow/underflow detection** for signed integer arithmetic operations. When overflow is detected, the program will **panic** with an "integer overflow" message.
 

### Arithmetic Operations Checked:
- **Addition** (`+`) - Detects positive and negative overflow
- **Subtraction** (`-`) - Detects underflow and overflow  
- **Multiplication** (`*`) - Detects overflow (with conservative checking)

### Integer Types Covered:
- `int8`, `int16`, `int32` (signed integers)
- **Note**: `int64`, `uintptr`, and unsigned types are **not checked** to avoid runtime compatibility issues

### Packages Excluded:
- Standard library packages (`runtime`, `sync`, `os`, `syscall`, etc.)
- Internal packages (`internal/*`)
- Math and unsafe packages

## Installation & Usage

### Usage and installation :
```bash
git clone https://github.com/kevin-valerio/go-arithmetic-panik

cd go-arithmetic-panik/src

./all.bash

# Compile and run a Go program
GOROOT=/path/to/go-arithmetic-panik/go-arithmetic-panik && ./bin/go run test_simple_overflow.go

# Or compile only
GOROOT=/path/to/go-arithmetic-panik/go-arithmetic-panik && ./bin/go build test_simple_overflow.go
```

## Example Overflow Detection

```go
package main

import "fmt"

func main() {
	fmt.Println("Testing overflow detection...")

	// Test int8 addition overflow
	var a int8 = 127
	var b int8 = 1
	fmt.Printf("Before: a=%d, b=%d\n", a, b)
	result := a + b  // Should panic with "integer overflow"
	fmt.Printf("After: result=%d\n", result)
}
```

Expected output:
```bash
bash-5.2$ GOROOT=/path/to/go-arithmetic-panik/go-arithmetic-panik && ./bin/go run test_simple_overflow.go
Testing overflow detection...
Before: a=127, b=1
panic: runtime error: integer overflow

goroutine 1 [running]:
main.main()
	/Users/XXX/go-arithmetic-panik/test_simple_overflow.go:12 +0xfc
exit status 2
```

## Limitations

1. **Standard Library**: No overflow checking in standard library code
2. **64-bit Integers**: `int64` and `uintptr` are not checked 
3. **Unsigned Types**: `uint8`, `uint16`, `uint32`, `uint64` are not checked
4. **Performance**: Slight performance overhead due to additional checks
