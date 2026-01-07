# cybergo: `go test -fuzz --use-libafl`

cybergo contains:

- `./`: a fork of the Go toolchain
- `golibafl/`: a LibAFL-based fuzzer that can fuzz Go code in-process via a libFuzzer-style entrypoint.

This repo adds a glue path so a user can keep writing **standard Go fuzz tests** (the ones used by `go test -fuzz=...`) and switch engines with a flag:

```bash
go test -fuzz=FuzzXxx --use-libafl
```

Without the flag, `go test -fuzz` behaves like upstream Go.

## Quick start

1) Build the forked toolchain:

```bash
cd src
./make.bash
```

2) Run a fuzz target with LibAFL (example in this repo):

```bash
cd test/cybergo/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --use-libafl
```

Fuzzing runs until you stop it (Ctrl+C). The run prints `ok ...` on clean shutdown.

## Requirements

- `CGO_ENABLED=1` (required; the harness exports C ABI symbols)
- Rust toolchain + `cargo` (because the runner is `golibafl`)
- Repo layout assumption: `golibafl/` must live inside the active `GOROOT` directory (at `$GOROOT/golibafl`).
  - In this repo that means `.../cybergo/golibafl`.

## What happens under the hood

When `go test` sees `-fuzz` + `--use-libafl`, it does **not** execute Go’s native fuzzing coordinator/worker engine.
Instead it builds a **libFuzzer-compatible harness** and runs the Rust LibAFL runner against it.

### 1) `go test` builds a libFuzzer harness archive (`libharness.a`)

For the test package’s generated main package (`_testmain.go`), cybergo also generates an extra file (`_libaflmain.go`) and switches the link mode:

- buildmode: `c-archive` → produces a static archive: `libharness.a`
- exports:
  - `LLVMFuzzerInitialize`
  - `LLVMFuzzerTestOneInput`

Those are the standard libFuzzer entrypoints expected by `libafl_targets::libfuzzer`.

`LLVMFuzzerInitialize` selects the fuzz target that matches the `-fuzz` regexp and initializes the captured Go fuzz callback.

`LLVMFuzzerTestOneInput` converts the input bytes and calls the fuzz callback once.

### 2) The fuzz callback is captured from standard `testing.F.Fuzz`

The `testing` package is patched so that, in this special mode, the first `f.Fuzz(func(*testing.T, []byte){...})` call is **captured** instead of starting the native Go fuzzing engine.

Then `testing.LibAFLFuzzOneInput` runs the captured callback in a normal `testing.T` context (using `tRunner`) so that `t.Fatal` / `t.FailNow` behave correctly (they call `runtime.Goexit`).

### 3) `go test` runs `golibafl` (Rust) with `HARNESS_LIB=...`

After building `libharness.a`, `go test` launches:

```bash
cargo run --release -- fuzz -i <input_dir> -o <output_dir>
```

with:

- `HARNESS_LIB=<path to built libharness.a>`
- working directory set to the `golibafl/` crate

`golibafl` was updated to accept `HARNESS_LIB` and **link the prebuilt archive** instead of rebuilding a harness from a `HARNESS=...` directory.

### 4) Output directories

cybergo reuses Go’s fuzz cache root (roughly `$(go env GOCACHE)/fuzz`).

For a package import path `example.com/mod/pkg`, LibAFL output goes under:

```
.../fuzz/example.com/mod/pkg/libafl/
  input/     # initial corpus dir (used if no testdata/fuzz exists; may be empty)
  queue/     # evolving corpus
  crashes/   # crashes (if any)
```

If the package has `testdata/fuzz/`, that directory is used as the initial `-i` corpus directory instead.
If the chosen `-i` directory is empty, `golibafl` generates a small random initial corpus.

## Current limitations

- Supports **multiple parameters** and cybergo’s fuzzable types:
  - all of Go’s native fuzzing primitives (`[]byte`, `string`, `bool`, `byte`, `rune`, `float32`, `float64`, `int*`, `uint*`)
  - composite types built from those (`struct`, `array`, `slice`, `*T`), recursively

```go
f.Fuzz(func(t *testing.T, in MyStruct, data []byte, n int) { ... })
```

## Notes / gotchas

- `--use-libafl` is intended only for `go test -fuzz=...`.
  It errors if you pass it without `-fuzz`.
- The toolchain adds the `libfuzzer` build tag during compilation in this mode, enabling Go’s `-d=libfuzzer` instrumentation path for coverage/cmp tracing.
