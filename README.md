# cybergo

[![cybergo --use-libafl smoke](https://github.com/kevin-valerio/cybergo/actions/workflows/smoke_use_libafl.yml/badge.svg?branch=master)](https://github.com/kevin-valerio/cybergo/actions/workflows/smoke_use_libafl.yml)
[![cybergo panikint self-compile and test](https://github.com/kevin-valerio/cybergo/actions/workflows/go.yml/badge.svg?branch=master)](https://github.com/kevin-valerio/cybergo/actions/workflows/go.yml)

cybergo is a security-focused fork of the Go toolchain.

It currently includes:
- `--use-libafl`: run standard `go test -fuzz=...` harnesses with LibAFL.
- Go-Panikint instrumentation: integer overflow/underflow detection for integer arithmetic, and optional integer truncation detection for integer conversions.

## LibAFL fuzzing

The goal of this fork is to integrate `go-libafl` (the Rust code in `golibafl/`) into the Go toolchain so you can run **standard Go fuzz tests** (`go test -fuzz=...`) with LibAFL by adding a flag:

```bash
CGO_ENABLED=1 ./bin/go test -fuzz=FuzzXxx --use-libafl
```

Without `--use-libafl`, `go test -fuzz` behaves like upstream Go.

### Quick start

#### Build the toolchain

Go requires a bootstrap Go toolchain. Set `GOROOT_BOOTSTRAP` to an existing Go install, then run:

```bash
cd src
GOROOT_BOOTSTRAP=/path/to/go ./make.bash
```

This produces `bin/go`.

#### Run a fuzz target with LibAFL

This mode requires `CGO_ENABLED=1` and a Rust toolchain (`cargo`), since the runner lives in `golibafl/`.

```bash
cd test/cybergo/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --use-libafl
```

#### More details

- Design notes: `misc/cybergo/USE_LIBAFL.md`
- Smoke tests: `misc/cybergo/tests/smoke_use_libafl.sh` (all) or `misc/cybergo/tests/smoke_use_libafl_*.sh` (per example)

### Examples

```bash
# Byte slice input
cd test/cybergo/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --use-libafl

# Struct input
cd test/cybergo/examples/multiargs
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzMultiArgs --use-libafl

# Multiple fuzzed parameters
cd test/cybergo/examples/multiparams
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzMultiParams --use-libafl
```

### Credits

Credits to Bruno Produit and Nills Ollrogge for their work on [golibafl](https://github.com/srlabs/golibafl/).

## Integer overflow & truncation detection (Go-Panikint)

This repo also includes the Go-Panikint compiler instrumentation (ported from the `trailofbits/go-panikint` fork).

### Overview

Go-Panikint adds **overflow/underflow detection** for integer arithmetic operations and **type truncation detection** for integer conversions. When overflow or truncation is detected, a **panic** with a detailed error message is triggered, including the specific operation type and integer types involved.

**Arithmetic operations**: Handles addition `+`, subtraction `-`, multiplication `*`, and division `/` for both signed and unsigned integer types. For signed integers, covers `int8`, `int16`, `int32`. For unsigned integers, covers `uint8`, `uint16`, `uint32`, `uint64`. The division case specifically detects the `MIN_INT / -1` overflow condition for signed integers. `int64` and `uintptr` are not checked for arithmetic operations.

**Type truncation detection**: Detects when integer type conversions would result in data loss due to the target type having a smaller range than the source type. Covers all integer types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`. Excludes `uintptr` due to platform-dependent usage. **Disabled** by default.

### Usage

Build the toolchain as usual (`cd src && ./make.bash`).

To enable truncation detection, rebuild with:

```bash
cd src
GOFLAGS="-gcflags=-truncationdetect=true" ./make.bash
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

### Testing

The Go-Panikint test suite lives in `tests/`:

```bash
cd tests
GOROOT=/path/to/cybergo /path/to/cybergo/bin/go test -v .
```

There is also a small fuzz harness in `fuzz_test/`:

```bash
cd fuzz_test
GOROOT=/path/to/cybergo /path/to/cybergo/bin/go test -fuzz=FuzzAdd -v
```

## About Go (upstream)

Go is an open source programming language that makes it easy to build simple,
reliable, and efficient software.

![Gopher image](https://golang.org/doc/gopher/fiveyears.jpg)
*Gopher image by [Renee French][rf], licensed under [Creative Commons 4.0 Attribution license][cc4-by].*

Upstream Go repository: https://go.googlesource.com/go.
Mirror: https://github.com/golang/go.

Unless otherwise noted, the Go source files are distributed under the
BSD-style license found in the LICENSE file.

## Download and Install

### Binary Distributions

Official binary distributions are available at https://go.dev/dl/.

After downloading a binary release, visit https://go.dev/doc/install
for installation instructions.

### Install From Source

If a binary distribution is not available for your combination of
operating system and architecture, visit
https://go.dev/doc/install/source
for source installation instructions.

## Contributing

Go is the work of thousands of contributors. We appreciate your help!

To contribute, please read the contribution guidelines at https://go.dev/doc/contribute.

Note that the Go project uses the issue tracker for bug reports and
proposals only. See https://go.dev/wiki/Questions for a list of
places to ask questions about the Go language.

[rf]: https://reneefrench.blogspot.com/
[cc4-by]: https://creativecommons.org/licenses/by/4.0/
