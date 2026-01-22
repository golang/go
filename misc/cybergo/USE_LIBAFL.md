# cybergo: `go test -fuzz --use-libafl`

cybergo contains:

- `./`: a fork of the Go toolchain
- `golibafl/`: a LibAFL-based fuzzer that can fuzz Go code in-process via a libFuzzer-style entrypoint.

This repo adds a glue path so a user can keep writing **standard Go fuzz tests** (the ones used by `go test -fuzz=...`) and switch engines with a flag:

```bash
go test -fuzz=FuzzXxx --use-libafl --focus-on-new-code=false
```

Without the flag, `go test -fuzz` behaves like upstream Go.

## Git-aware scheduling (focus on new code)

When `--use-libafl` is set, `--focus-on-new-code={true|false}` is **required**.

- `--focus-on-new-code=false`: keep the current behavior.
- `--focus-on-new-code=true`: prefer inputs that execute recently changed lines (based on `git blame`).

Note: `--focus-on-new-code=true` needs `git` (to run `git blame`) and an `addr2line` implementation to map coverage counters back to source `file:line` (prefer `llvm-addr2line`; fall back to binutils `addr2line`).

Implementation note: the git-aware scheduler currently comes from a local LibAFL fork (TODO: switch back to upstream LibAFL once upstreamed).

### Benchmark (geth)

A paired benchmark for `--focus-on-new-code` on a shallow clone of go-ethereum (geth) lives at `misc/cybergo/bench_focus_on_new_code_geth.sh`.

## Runner configuration

cybergo can pass a JSONC configuration file (JSON with `//` comments) to the LibAFL runner:

```bash
go test -fuzz=FuzzXxx --use-libafl --focus-on-new-code=false --libafl-config=libafl.jsonc
```

`golibafl` also needs a TCP broker port for LibAFL's internal event manager. By default, it picks a **random free port** (instead of always `1337`). If you need a fixed port, set `GOLIBAFL_BROKER_PORT=1337` (or pass `-p/--port 1337` when running `golibafl` directly).

Example `libafl.jsonc` (all fields optional; defaults shown in comments):

```jsonc
{
  // cores: CPU cores to bind LibAFL clients to (ex: "0,1" / "all" / "none")
  // default (cybergo go test): "0" (single client)
  "cores": "0,1",

  // exec_timeout_ms: per-execution timeout for the in-process harness
  // default: 1000
  "exec_timeout_ms": 1000,

  // corpus_cache_size: in-memory cache size for each on-disk corpus
  // default: 4096
  "corpus_cache_size": 4096,

  // initial_generated_inputs: generated corpus size if the input dir is empty
  // default: 8
  "initial_generated_inputs": 8,

  // initial_input_max_len: max length for generated initial inputs
  // default: 32
  "initial_input_max_len": 32,

  // tui_monitor: enable LibAFL's interactive terminal UI (TUI)
  // default: true
  "tui_monitor": true,

  // debug_output: force-enable/disable LIBAFL_DEBUG_OUTPUT (otherwise auto)
  // default: auto (enabled when running with a single client)
  "debug_output": true
}
```

A ready-to-edit template lives at `misc/cybergo/libafl.config.jsonc`.

## Troubleshooting

If `golibafl` fails to launch, set `CYBERGO_VERBOSE_AFL=1` to print extra diagnostics and write them to `OUTPUT_DIR/golibafl_launcher_failure_<pid>.txt`.

If fuzzing prints repeated timeouts with **0 executions** or appears stuck during startup, make sure you are using a LibAFL fork/build that runs the restarting manager in **non-fork (re-exec) mode**. The embedded Go runtime is not fork-safe once initialized, so forking-based restarts can deadlock and look like “exec/sec: 0.000”.

## Quick start

1) Build the forked toolchain:

```bash
cd src
./make.bash
```

2) Run a fuzz target with LibAFL (example in this repo):

```bash
cd test/cybergo/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --use-libafl --focus-on-new-code=false
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
- `HARNESS_LINK_SEARCH=/path/one:/path/two` (optional: extra native link search dirs)
- `HARNESS_LINK_LIBS=dylib=dl,static=z` (optional: extra `rustc-link-lib` entries)
- working directory set to the `golibafl/` crate

`golibafl` was updated to accept `HARNESS_LIB` and **link the prebuilt archive** instead of rebuilding a harness from a `HARNESS=...` directory.

### 4) Output directories

cybergo reuses Go’s fuzz cache root (roughly `$(go env GOCACHE)/fuzz`) so `go clean -fuzzcache` works.

LibAFL output is separated per **project + package + fuzz target**:

```
.../fuzz/<pkg import path>/libafl/<project>/<harness>/
  input/     # initial corpus dir (merged from testdata/fuzz + f.Add); may be empty
  queue/     # evolving corpus
  crashes/   # crashes (if any)
```

- `<project>` is derived from the package root directory (module root / GOPATH / GOROOT root) and formatted as `<basename>-<hash>`.
- `<harness>` is the fuzz target name when `-fuzz` is a simple identifier like `FuzzXxx` (or `^FuzzXxx$`), otherwise `pattern-<hash>`.

On each run, cybergo prepares `<...>/input/` as the initial `-i` corpus directory:

- files from `testdata/fuzz/` (if it exists) are copied into it
- manual seeds provided via `f.Add(...)` are written into it automatically
- if this fuzzing campaign was already run before, the previous LibAFL `queue/` corpus is automatically reused on restart (so Ctrl-C + rerun continues from the same corpus by default)

If the chosen `-i` directory is empty, `golibafl` generates a small random initial corpus.

On shutdown, `go test` prints the full output directory path:

```
libafl output dir: /full/path/to/.../libafl/<project>/<harness>
```

### Crash handling (stop on first crash)

In `--use-libafl` mode, cybergo follows `go test -fuzz` semantics: **stop the whole fuzzing run on the first crash** (even with multiple LibAFL clients).

When a crash is found, `golibafl` prints:
- the output directory + `crashes/` path
- the exact crash input file path
- a repro command: `golibafl run --input <crash_file>`

To keep fuzzing after crashes, set `"stop_all_fuzzers_on_panic": false` in the LibAFL JSONC config.

Note: reproducing may require the same runtime environment variables as fuzzing (e.g. `LD_LIBRARY_PATH` for native deps).

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
- On Unix, `golibafl` uses LibAFL’s restarting manager in **non-fork (re-exec) mode** for reliability with the embedded Go runtime. This is also why `golibafl` switches its working directory to `OUTPUT_DIR/workdir/<pid>` (so respawns don’t fail if the original cwd is deleted/unlinked).
