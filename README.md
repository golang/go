# cybergo

[![cybergo --use-libafl smoke](https://github.com/kevin-valerio/cybergo/actions/workflows/smoke_use_libafl.yml/badge.svg?branch=master)](https://github.com/kevin-valerio/cybergo/actions/workflows/smoke_use_libafl.yml)
[![cybergo panikint self-compile and test](https://github.com/kevin-valerio/cybergo/actions/workflows/go.yml/badge.svg?branch=master)](https://github.com/kevin-valerio/cybergo/actions/workflows/go.yml)

cybergo is a security-focused fork of the Go toolchain. In a _very_ simple phrasing, cybergo is a copy of the Go compiler that finds bugs. For now, it focuses on two things:

- Integrating [go-panikint](https://github.com/trailofbits/go-panikint): instrumentation that panics on **integer overflow/underflow** (and **optionally on truncating integer conversions**).
- Integrating [LibAFL](https://github.com/AFLplusplus/LibAFL) fuzzer : run Go fuzzing harnesses with **LibAFL** for better fuzzing performances.

## Build
```bash
cd src && ./make.bash # this produces `./bin/go`
```

## Feature 1: integer overflow and truncation issues detection

#### Overview

This work is inspired from the previously developed [go-panikint](https://github.com/trailofbits/go-panikint). It adds overflow/underflow detection for integer arithmetic operations and (optionnally) type truncation detection for integer conversions. When overflow or truncation is detected, a **panic** with a detailed error message is triggered, including the specific operation type and integer types involved.

Arithmetic operations: Handles addition `+`, subtraction `-`, multiplication `*`, and division `/` for both signed and unsigned integer types. For signed integers, covers `int8`, `int16`, `int32`. For unsigned integers, covers `uint8`, `uint16`, `uint32`, `uint64`. The division case specifically detects the `MIN_INT / -1` overflow condition for signed integers. `int64` and `uintptr` are not checked for arithmetic operations.

Type truncation detection: Detects potentially lossy integer type conversions. Covers all integer types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`. Excludes `uintptr` due to platform-dependent usage. This is disabled by default.

#### How it works

This feature patches the compiler SSA generation so that integer arithmetic operations and integer conversions get extra runtime checks that call into the runtime to panic with a detailed error message when a bug is detected. Checks are applied using source-location-based filtering so user code is instrumented while standard library files and dependencies (module cache and `vendor/`) are skipped.

You can read the associated blog post about it [**here**](https://blog.trailofbits.com/2025/12/31/detect-gos-silent-arithmetic-bugs-with-go-panikint/).

#### Suppressing false positives

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


## Feature 2: LibAFL

Using the `--use-libafl` flag runs standard Go fuzz tests (`go test -fuzz=...`) **with** [LibAFL](https://github.com/AFLplusplus/LibAFL). The runner is implemented in `golibafl/`.

```bash
./bin/go test -fuzz=FuzzXxx --use-libafl
```
Without `--use-libafl`, `go test -fuzz` behaves like upstream Go.

#### LibAFL (Go) benchmarks

LibAFL performs *way* better than the traditional Go fuzzer.


##### Benchmark 1:

The chart below is the evolution of the number of lines covered while fuzzing Google's [UUID](https://github.com/google/uuid) using LibAFL vs go compiler.
![BENCH1](misc/cybergo/5min_uuid_parsebytes_FuzzParseBytes.png "BENCH1")


#### Example
You can test it on some fuzzing harnesses in `test/cybergo/examples/`.

```bash
cd test/cybergo/examples/reverse
../../../../bin/go test -fuzz=FuzzReverse --use-libafl
```

## Credits

Credits to Bruno Produit and Nills Ollrogge for their work on [golibafl](https://github.com/srlabs/golibafl/).
