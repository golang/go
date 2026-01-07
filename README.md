# cybergo

[![cybergo --use-libafl smoke](https://github.com/kevin-valerio/cybergo/actions/workflows/smoke_use_libafl.yml/badge.svg?branch=master)](https://github.com/kevin-valerio/cybergo/actions/workflows/smoke_use_libafl.yml)
[![cybergo panikint self-compile and test](https://github.com/kevin-valerio/cybergo/actions/workflows/go.yml/badge.svg?branch=master)](https://github.com/kevin-valerio/cybergo/actions/workflows/go.yml)

cybergo is a security-focused fork of the Go toolchain.

The goal of this fork is to be security oriented. For now, it focuses on two things:

- **Go-Panikint**: compiler instrumentation that panics on integer overflow/underflow (and optionally on truncating integer conversions).
- **LibAFL fuzzing** (`--use-libafl`): run standard `go test -fuzz=...` harnesses with LibAFL.

## Build

Go requires a bootstrap Go toolchain. Set `GOROOT_BOOTSTRAP` to an existing Go install, then run:

```bash
cd src
GOROOT_BOOTSTRAP=/path/to/go ./make.bash
```

This produces `bin/go`.

## Go-Panikint

### Overview

Go-Panikint adds **overflow/underflow detection** for integer arithmetic operations and **type truncation detection** for integer conversions. When overflow or truncation is detected, a **panic** with a detailed error message is triggered, including the specific operation type and integer types involved.

**Arithmetic operations**: Handles addition `+`, subtraction `-`, multiplication `*`, and division `/` for both signed and unsigned integer types. For signed integers, covers `int8`, `int16`, `int32`. For unsigned integers, covers `uint8`, `uint16`, `uint32`, `uint64`. The division case specifically detects the `MIN_INT / -1` overflow condition for signed integers. `int64` and `uintptr` are not checked for arithmetic operations.

**Type truncation detection**: Detects potentially lossy integer type conversions. Covers all integer types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`. Excludes `uintptr` due to platform-dependent usage. **Disabled** by default.

### How it works

Go-Panikint patches the compiler SSA generation so that integer arithmetic operations and integer conversions get extra runtime checks that call into the runtime to panic with a detailed error message when a bug is detected. Checks are applied using source-location-based filtering so user code is instrumented while standard library files and dependencies (module cache and `vendor/`) are skipped.

### Enabling truncation detection

Truncation detection is controlled by a compiler flag. Enable it for a build/test with:

```bash
./bin/go test -gcflags=all=-truncationdetect=true ./...
```

### Suppressing false positives

Add a marker on the same line as the operation or the line immediately above to suppress a specific report:

- Overflow/underflow: `overflow_false_positive`
- Truncation: `truncation_false_positive`

Example:

```go
// overflow_false_positive
intentionalOverflow := a + b

// truncation_false_positive
x := uint8(big)

sum2 := a + b // overflow_false_positive
x2 := uint8(big) // truncation_false_positive
```

Test vectors live in `tests/` and `fuzz_test/`.

## LibAFL fuzzing

`--use-libafl` runs **standard Go fuzz tests** (`go test -fuzz=...`) with LibAFL. The runner is implemented in `golibafl/`.

This mode requires `CGO_ENABLED=1` and a Rust toolchain (`cargo`).

```bash
CGO_ENABLED=1 ./bin/go test -fuzz=FuzzXxx --use-libafl
```

Without `--use-libafl`, `go test -fuzz` behaves like upstream Go.

### Example

```bash
cd test/cybergo/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --use-libafl
```

More examples live in `test/cybergo/examples/`.

## Credits

- Credits to Bruno Produit and Nills Ollrogge for their work on [golibafl](https://github.com/srlabs/golibafl/).
