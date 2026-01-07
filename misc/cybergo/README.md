# CyberGo

CyberGo is a security-focused fork of the Go toolchain that adds an opt-in path to run **standard Go fuzz tests** (`go test -fuzz=...`) using the LibAFL-based runner in `golibafl/`.

## Quick start

### Build the toolchain

Go requires a bootstrap Go toolchain. Set `GOROOT_BOOTSTRAP` to an existing Go install (Go >= 1.24.6), then run:

```bash
cd src
GOROOT_BOOTSTRAP=/path/to/go ./make.bash
```

This produces `bin/go`.

### 2) Run a fuzz target with LibAFL


```bash
cd test/cybergo/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --use-libafl
```

## Test me

Byte array:

cd test/cybergo/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --use-libafl

Struct:

cd test/cybergo/examples/multiargs
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzMultiArgs --use-libafl

Multiple fuzzed parameters:

cd test/cybergo/examples/multiparams
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzMultiParams --use-libafl

## More details

See `misc/cybergo/USE_LIBAFL.md`.

## Smoke test

Run `misc/cybergo/tests/smoke_use_libafl.sh` to build the toolchain and run a deterministic crashing fuzz target under `--use-libafl`.
