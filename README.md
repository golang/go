# CyberGo

[![CyberGo --use-libafl smoke](https://github.com/kevin-valerio/cybergo/actions/workflows/smoke_use_libafl.yml/badge.svg?branch=master)](https://github.com/kevin-valerio/cybergo/actions/workflows/smoke_use_libafl.yml)

CyberGo is a security-focused fork of the Go toolchain.

The goal of this fork is to integrate `go-libafl` (the Rust code in `golibafl/`) into the Go toolchain so you can run **standard Go fuzz tests** (`go test -fuzz=...`) with LibAFL by adding a flag:

```bash
CGO_ENABLED=1 ./bin/go test -fuzz=FuzzXxx --use-libafl
# (alias) CGO_ENABLED=1 ./bin/go test -fuzz=FuzzXxx --use-golibafl
```

Without `--use-libafl`, `go test -fuzz` behaves like upstream Go.

## Quick start

### Build the toolchain

Go requires a bootstrap Go toolchain. Set `GOROOT_BOOTSTRAP` to an existing Go install, then run:

```bash
cd src
GOROOT_BOOTSTRAP=/path/to/go ./make.bash
```

This produces `bin/go`.

### Run a fuzz target with LibAFL

This mode requires `CGO_ENABLED=1` and a Rust toolchain (`cargo`), since the runner lives in `golibafl/`.

```bash
cd test/cybergo/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --use-libafl
```

### More details

- Design notes: `misc/cybergo/USE_LIBAFL.md`
- Smoke tests: `misc/cybergo/tests/smoke_use_libafl.sh` (all) or `misc/cybergo/tests/smoke_use_libafl_*.sh` (per example)

## Examples

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
