# gosentry: `go test -fuzz` (LibAFL by default)

gosentry contains:

- `./`: a fork of the Go toolchain
- `golibafl/`: a LibAFL-based fuzzer that can fuzz Go code in-process via a libFuzzer-style entrypoint.

This repo adds a glue path so a user can keep writing **standard Go fuzz tests** (the ones used by `go test -fuzz=...`) and switch engines.

By default, when `-fuzz` is set, gosentry uses the LibAFL runner.

```bash
go test -fuzz=FuzzXxx --focus-on-new-code=false --catch-races=false --catch-leaks=false
```

Stability note: gosentry sets `GODEBUG=updatemaxprocs=0` automatically for LibAFL fuzzing.
This disables the runtime's automatic GOMAXPROCS updates, which can trigger an intermittent
`fatal error: sync: inconsistent mutex state` on Linux CI.

Stop the fuzz campaign with Ctrl+C.

Use `-fuzztime=<duration>` to run a bounded LibAFL campaign:

```bash
go test -fuzz=FuzzXxx -fuzztime=1m --focus-on-new-code=false --catch-races=false --catch-leaks=false
```

LibAFL mode supports the duration form, like `30s` or `1m`. The upstream count form, like `1000x`, is not supported in LibAFL mode.

To opt out:

```bash
go test -use-libafl=false -fuzz=FuzzXxx
```

## Generate coverage from a LibAFL campaign (HTML)

After (or while) running a LibAFL fuzz campaign, gosentry can generate a Go coverage report by replaying the current LibAFL **queue corpus**.

```bash
# Same package + same fuzz target as your fuzz campaign:
go test -fuzz=FuzzXxx --generate-coverage .
```

Notes:
- Use the same `-fuzz` spelling you used for the campaign you want to replay. gosentry uses `-fuzz` to locate the LibAFL output directory (and its `queue/` corpus).
  - `-fuzz=FuzzXxx` and `-fuzz='^FuzzXxx$'` refer to the same campaign.
  - If you fuzzed with a broader regexp (example: `-fuzz='Fuzz.*Parser'`), reuse the exact same regexp for coverage.
- `-fuzz` must match exactly one fuzz target, otherwise gosentry can’t select which campaign to replay.

This replays inputs from `<libafl output dir>/queue/` and writes:
- `<libafl output dir>/coverage/cover.out` (Go coverprofile format)
- `<libafl output dir>/coverage/cover.html` (HTML report)

At the end, gosentry prints the full paths to both files.

Note: this does **not** run fuzzing. For large corpora, consider `-timeout=0`.

## Grammar-based fuzzing (Nautilus)

When running in LibAFL mode, gosentry can generate and mutate inputs using Nautilus, a grammar-based mutator integrated in LibAFL. This is useful for parsers and file formats where byte-level mutation wastes time on syntactically-invalid inputs.

Flags (gosentry `go test`):
- `--use-grammar`: enable grammar-based input generation.
  - Requires `-fuzz=...` and LibAFL mode.
  - Still requires the usual LibAFL-required flags: `--focus-on-new-code={true|false}`, `--catch-races={true|false}`, `--catch-leaks={true|false}`.
- `--grammar path/to/grammar.json`: Nautilus JSON grammar file.
  - Must be specified exactly once.
  - Start symbol is implicit: the first rule’s nonterminal in the JSON file.

Tuning:
- In `--libafl-config`, set:
  - `nautilus_max_len` (only used with `--use-grammar`).
  - `nautilus_cmplog_i2s` (default: true). Set to `false` to disable the grammar CMPLOG/I2S stage.

Requirements:
- No extra dependencies beyond the Rust toolchain already needed for LibAFL mode.

<details>
<summary><strong>Command example</strong></summary>

```bash
# Example (from this repo): JSON grammar + JSON harness.
cd test/gosentry/examples/grammar_json
GOSENTRY_VERBOSE_AFL=1 CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzGrammarJSON \
  --use-grammar --grammar=testdata/JSON.json \
  --focus-on-new-code=false --catch-races=false --catch-leaks=false .

# Generic pattern:
# GOSENTRY_VERBOSE_AFL=1 CGO_ENABLED=1 go test -fuzz=FuzzXxx \
#   --use-grammar --grammar=/path/to/MyGrammar.json \
#   --focus-on-new-code=false --catch-races=false --catch-leaks=false
```

</details>

<details>
<summary><strong>Go fuzz harness example (JSON)</strong></summary>

```go
package mypkg

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"
)

func FuzzGrammarJSON(f *testing.F) {
	f.Fuzz(func(t *testing.T, data []byte) {
		dec := json.NewDecoder(bytes.NewReader(data))
		dec.UseNumber()

		var v any
		if err := dec.Decode(&v); err != nil {
			t.Fatalf("invalid JSON: %v", err)
		}
		if err := dec.Decode(&struct{}{}); err != io.EOF {
			t.Fatalf("invalid JSON: trailing data")
		}
	})
}
```

</details>

<details>
<summary><strong>Nautilus JSON grammar example (small JSON subset)</strong></summary>

This is the file format expected by `--grammar=...`:
- The grammar is a JSON array of rules: `["NonTerm", "RHS"]`.
- Nonterminal names must start with a capital letter (`Value`, `Object`, ...).
- Use `{NonTerm}` in the RHS to reference another rule.
- `{` and `}` are reserved for nonterminal references; to emit literal braces, use `\\{` and `\\}` in the RHS string.

```json
[
  ["Json", "{Value}"],
  ["Value", "null"],
  ["Value", "{String}"],
  ["String", "\"{Chars}\""],
  ["Chars", ""],
  ["Chars", "{Char}{Chars}"],
  ["Char", "a"],
  ["Char", "b"]
]
```

</details>

<details>
<summary><strong>Expected verbose output (example)</strong></summary>

```text
golibafl: nautilus enabled (grammar=... max_len=... workdir=...)
GOLIBAFL_MUTATED_INPUT "null"
GOLIBAFL_MUTATED_INPUT "{}"
GOLIBAFL_MUTATED_INPUT "[true,false,null]"
```

</details>

Set `GOSENTRY_VERBOSE_AFL=1` to print a few generated inputs as `GOLIBAFL_MUTATED_INPUT "..."` (printed by `golibafl`, useful for smoke tests / CI). Set `GOSENTRY_VERBOSE_AFL_ALL_INPUTS=1` to print `GOLIBAFL_MUTATED_INPUT` for **every** grammar-mode execution (very noisy).

Behavior notes:
- If the LibAFL input dir is empty, `golibafl` generates an initial corpus using the grammar.
- If you provide initial seeds (via `testdata/fuzz` or by placing files in the LibAFL input dir), they will be loaded into the corpus and Nautilus will mutate the selected corpus seed (coverage-guided) instead of overwriting it with unrelated fresh generations.
- If a loaded corpus seed is not parseable by the grammar, `golibafl` falls back to generation-from-scratch instead of aborting the fuzz run.
- In grammar mode, `golibafl` keeps inputs grammar-valid: it uses Nautilus mutation plus (by default) a CMPLOG-guided, I2S-like stage that rewrites Nautilus leaf terminals (no raw byte-level havoc/token stages), so the corpus stays parseable by the grammar. You can disable that stage with `--libafl-config` by setting `nautilus_cmplog_i2s=false` (byte-level fuzzing still keeps CMPLOG/I2S always on).
- The `GOLIBAFL_MUTATED_INPUT` log is capped to the first 20 executions by default; set `GOSENTRY_VERBOSE_AFL_ALL_INPUTS=1` to remove the cap.

Limitations (current glue):
- Grammar mode works best with a single input argument; multi-arg fuzz targets will decode the underlying byte buffer into separate values.
- Grammar mode has no two-seed crossover/recombination (mutation is single-seed).

### CI note

This repo includes a grammar smoke test script at `misc/gosentry/tests/smoke_use_libafl_grammar_json.sh` (run by `.github/workflows/smoke_use_libafl.yml`).
It also includes a CMPLOG/I2S integration smoke test for grammar mode at `misc/gosentry/tests/smoke_use_libafl_grammar_cmplog_i2s.sh`.

<details>
<summary><strong>How grammar fuzzing works in gosentry (detailed)</strong></summary>

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ 0) gosentry `go test -fuzz=FuzzXxx` (LibAFL + --use-grammar)               │
│    - captures your fuzz target + signature (`testing/libafl.go`)            │
│    - builds `libharness.a` (buildmode=c-archive)                            │
│    - runs the Rust runner: `golibafl fuzz ... --use-grammar ...`            │
└───────────────┬───────────────────────────────────────────────────────────┘
                v
┌───────────────────────────────────────────────────────────────────────────┐
│ 1) `golibafl` (Rust + LibAFL) fuzzes the Go harness in-process              │
│    - loads `libharness.a` via `HARNESS_LIB=...`                             │
│    - executes inputs by calling `LLVMFuzzerTestOneInput(data)`              │
│    - scheduler selects a corpus seed (coverage-guided)                      │
│    - stage mutates that seed, executes, and keeps new-coverage inputs       │
│      in the on-disk corpus (`output/queue/`)                                │
└───────────────┬───────────────────────────────────────────────────────────┘
                v
┌───────────────────────────────────────────────────────────────────────────┐
│ 2) Nautilus (in-process, per client)                                        │
│    - loads the JSON grammar into a Nautilus context                          │
│    - if the input dir is empty: generates an initial corpus                  │
│    - fuzz loop stage: parse seed -> mutate tree -> unparse to bytes          │
│    - if the seed is not parseable: fall back to generation-from-scratch      │
└───────────────────────────────────────────────────────────────────────────┘
```

Notes:
- Your Go harness still receives standard Go fuzz inputs. In grammar mode, a one-arg fuzz target can be either `data []byte` or `s string` (the generated sample is passed as UTF-8 bytes).
- Grammar mode works best with a single input argument; with multiple arguments, gosentry will decode the underlying byte buffer into separate values, so the original grammar-generated text won’t stay intact.
- If the harness rejects inputs (example: JSON unmarshal fails), it usually means the grammar is not aligned with what the harness expects.
- Use `GOSENTRY_VERBOSE_AFL=1` to see `golibafl: nautilus enabled` + a few `GOLIBAFL_MUTATED_INPUT "..."` lines (or `GOSENTRY_VERBOSE_AFL_ALL_INPUTS=1` for all of them).

</details>

## Git-aware scheduling (focus on new code)

`--focus-on-new-code={true|false}` enables git-aware scheduling to prefer inputs that execute recently changed lines (based on `git blame`).

This flag only applies in LibAFL mode and is required.

- `--focus-on-new-code=false`: keep the current behavior.
- `--focus-on-new-code=true`: prefer inputs that execute recently changed lines (based on `git blame`).

Note: `--focus-on-new-code=true` needs `git` (to run `git blame`) and `go tool addr2line` to map coverage counters back to source `file:line`.

## Catching data races (optional)

`--catch-races={true|false}` enables a separate `-race` replay loop that watches the LibAFL `queue/` corpus directory and replays only newly discovered seeds with `GORACE=halt_on_error=1`.

When a race is detected, gosentry prints the race detector report (stack traces), prints the exact seed path, copies the seed into `output/races/` (under the LibAFL output directory), and stops the fuzz campaign (treats it like a bug/crash).

This flag only applies in LibAFL mode and is required.

Note: the Go race detector only reports unsynchronized concurrent access between goroutines in a single process. If the target has no concurrency / shared mutable state during the replay run, there may be no races to catch.

Linux note: the race replay runner must be linked as a **non-PIE** executable. If you see `ThreadSanitizer failed to allocate ...` during replay, it usually means the runner was built as PIE; rebuild it with `-no-pie`.

### CI note

This repo runs the `--catch-races` smoke test in its own GitHub Actions workflow (`.github/workflows/catch_races.yml`) so it shows up as a separate check.

## Catching goroutine leaks (optional)

`--catch-leaks={true|false}` enables a `goleak` replay loop that watches the LibAFL `queue/` corpus directory and replays only newly discovered seeds with `go.uber.org/goleak`.

When a goroutine leak is detected, gosentry prints the exact seed path, copies the seed into `output/leaks/` (under the LibAFL output directory), and stops the fuzz campaign (treats it like a bug/crash).

This flag only applies in LibAFL mode and is required.

Note: `goleak` is for **goroutine leaks**, not memory leaks.

### CI note

This repo runs the `--catch-leaks` smoke test in its own GitHub Actions workflow (`.github/workflows/catch_leaks.yml`) so it shows up as a separate check.

## Catching confirmed hangs (optional)

When fuzzing with LibAFL, a harness execution may **timeout** (for example because of a deadlock / goroutines waiting forever, or an extremely slow path).

gosentry can treat a timeout as a **hang candidate**, replay it a few times with a larger timeout to confirm it is a **definitive hang**, then stop the fuzz campaign like a bug/crash.

Note: hang confirmation also runs during initial corpus import/generation, so targets that time out on every input can still be detected deterministically.

This is configured via the LibAFL runner JSONC config:

- `catch_hangs` (default: `true`): enable/disable hang confirmation.
- `hang_timeout_ms` (default: `10000`): wall-clock timeout for each confirmation replay.
- `hang_confirm_runs` (default: `3`): number of confirmation replays.

On a confirmed hang, `golibafl` writes the input to `output/hangs/` (under the LibAFL output directory) and stops the fuzz campaign when `stop_all_fuzzers_on_panic=true`.

Before exiting, `golibafl` attempts to minimize the crashing/hanging input (best-effort; hangs are capped to ~60s total).

`golibafl` stores the generated mapping file under the LibAFL output directory as `git_recency_map.bin` (path provided to the runner via `LIBAFL_GIT_RECENCY_MAPPING_PATH`).
On large targets, generating this file can take several minutes because it needs to run `go tool addr2line` and `git blame` for many coverage counters.

To make this less “silent”, `golibafl` prints progress during generation (and runs `go tool addr2line` in parallel):

```
golibafl: generating git recency mapping (450229 counters); this may take a while
golibafl: running go tool addr2line on 450229 addresses
golibafl: addr2line sent 120000/450229 got 80000/450229 elapsed 30s
golibafl: running git blame on 1043 file(s)
golibafl: git blame progress 702/1043 (67.3%), elapsed 20s
```

To avoid regenerating the full mapping on commits that don't change the Go harness object (`go.o`), `golibafl` writes a small sidecar file next to it: `git_recency_map.bin.meta.json`.
If the sidecar indicates the current `go.o` hash matches, `golibafl` will reuse the existing mapping entries and only update the `head_time` header in `git_recency_map.bin`.

Implementation note: the git-aware scheduler currently comes from a local LibAFL fork (TODO: switch back to upstream LibAFL once upstreamed).

### Benchmark (geth)

A paired benchmark for `--focus-on-new-code` on a shallow clone of go-ethereum (geth) lives at `misc/gosentry/bench_focus_on_new_code_geth.sh`.

## Runner configuration

gosentry can pass a JSONC configuration file (JSON with `//` comments) to the LibAFL runner:

```bash
go test -fuzz=FuzzXxx --focus-on-new-code=false --catch-races=false --catch-leaks=false --libafl-config=libafl.jsonc
```

`golibafl` also needs a TCP broker port for LibAFL's internal event manager. By default, it picks a **random free port** (instead of always `1337`). If you need a fixed port, set `GOLIBAFL_BROKER_PORT=1337` (or pass `-p/--port 1337` when running `golibafl` directly).

Implementation note: `golibafl` configures LibAFL's broker to exit cleanly after the **first** client has connected (instead of waiting for all configured cores to connect). This avoids hangs in multi-client runs if a client exits early (or never connects) during shutdown.

Example `libafl.jsonc` (all fields optional; defaults shown in comments):

```jsonc
{
  // cores: CPU cores to bind LibAFL clients to (ex: "0,1" / "all" / "none")
  // default (gosentry go test): "0" (single client)
  "cores": "0,1",

  // exec_timeout_ms: per-execution timeout for the in-process harness
  // default: 1000
  "exec_timeout_ms": 1000,

  // catch_hangs: confirm per-execution timeouts as definitive hangs by replaying the timed-out input
  // default: true
  "catch_hangs": true,

  // hang_timeout_ms: wall-clock timeout for hang confirmation replays (`golibafl run --input ...`)
  // default: 10000
  "hang_timeout_ms": 10000,

  // hang_confirm_runs: number of confirmation replays for a timed-out input
  // default: 3
  "hang_confirm_runs": 3,

  // git_recency_alpha: bias strength for git-aware scheduling when --focus-on-new-code=true
  // min: 0.0
  // max: 10.0
  // default: 2.0
  // recommended: 2.0
  "git_recency_alpha": 2.0,

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
  // default: true (auto-disabled when not in the terminal foreground process group, to avoid getting stopped by job control)
  "tui_monitor": true,

  // debug_output: force-enable/disable LIBAFL_DEBUG_OUTPUT (otherwise auto)
  // default: auto (enabled when running with a single client)
  "debug_output": true,

  // --- Grammar fuzzing (Nautilus) ---
  // Only used with -use-grammar (or golibafl --use-grammar).

	  // nautilus_max_len: maximum expansion length used by Nautilus when generating/mutating
	  // default: 64
	  // example: 256 (allow deeper nesting and longer strings)
	  "nautilus_max_len": 64,

	  // nautilus_cmplog_i2s: enable/disable the grammar CMPLOG/I2S stage
	  // default: true
	  "nautilus_cmplog_i2s": true
	}
```

A ready-to-edit template lives at `misc/gosentry/libafl.config.jsonc`.

## Troubleshooting

If `golibafl` fails to launch, set `GOSENTRY_VERBOSE_AFL=1` to print extra diagnostics and write them to `OUTPUT_DIR/golibafl_launcher_failure_<pid>.txt`.

If fuzzing "hangs" right after printing `Running target/release/golibafl fuzz ...` and `ps`/`htop` shows `golibafl` in state `T` (stopped), it's usually the TUI monitor getting stopped by the terminal (job control). This happens because `go test` runs `golibafl` in a separate process group. `golibafl` auto-disables the TUI in this situation and falls back to the simple monitor.

If fuzzing prints repeated timeouts with **0 executions** or appears stuck during startup, make sure you are using a LibAFL fork/build that runs the restarting manager in **non-fork (re-exec) mode**. The embedded Go runtime is not fork-safe once initialized, so forking-based restarts can deadlock and look like “exec/sec: 0.000”.

If you see a panic like `BUG: The current message never got committed using send!` (from `crates/ll_mp/...`), it's a LibAFL/LLMP issue (not a Go harness bug). `golibafl` enables OOM-safe restart-state serialization so the respawner can recover from unexpected child exits without panicking; if you still hit it, rebuild/update `golibafl` and retry (and consider setting `RUST_BACKTRACE=1` to capture a backtrace).

If you see a panic like `called Option::unwrap() on a None value` in `crates/libafl/src/corpus/inmemory.rs` (often while inserting into the corpus), it's a LibAFL `InMemoryCorpus` internal bookkeeping issue. gosentry enables LibAFL's `corpus_btreemap` feature in `golibafl` to avoid the affected code path; rebuild `golibafl` and retry.

If you see a panic like `The testcase is not associated with an id`, rebuild `golibafl` with the current lockfile. The pinned LibAFL fork sets `corpus_id` for both normal and explicit-ID `corpus_btreemap` insertions, as required by `CachedOnDiskCorpus` when it loads inputs from disk.

## Maintainer notes

- `golibafl/build.rs` derives the `-l static=...` library name from `HARNESS_LIB` (for example `libharness_race.a` becomes `-l static=harness_race`). This matters for `--catch-races`, which builds `libharness_race.a` in a separate directory on CI/Linux.
- `golibafl/build.rs`: `built_harness` is now declared only on macOS (`cfg(target_os = "macos")`) to avoid `unused variable` warnings on non-macOS targets (no behavior change).

### Documentation notes

- `README.md` now includes short flow diagrams for `--catch-races` and `--catch-leaks` (Feature 5), including when the dedicated replay runner is built, where it is copied, and how the queue monitoring works.

### Replay-only harness builds

`--catch-races` needs a separate `-race` harness archive (`libharness_race.a`) to replay seeds with the Go race detector.
That harness is built via a nested `go test` subprocess with `GOSENTRY_LIBAFL_BUILD_ONLY=1`.

When `GOSENTRY_LIBAFL_BUILD_ONLY=1`, gosentry intentionally disables fuzz coverage instrumentation (`-d=libfuzzer`) for the harness build.
The replay runner only needs to execute inputs, not measure coverage, and keeping the `-race` replay harness uninstrumented avoids mixing the fuzzing coverage instrumentation with the race detector build.

### Sidecar queue snapshot (catch-races / catch-leaks)

The `--catch-races` and `--catch-leaks` sidecars decide what is "new" by keeping a `seen` set of seed paths under `output/queue/`.

To avoid flaky behavior (missing very fast early seeds due to goroutine scheduling), gosentry snapshots the initial contents of `output/queue/` **before** starting the main LibAFL fuzz campaign, and uses that as the initial `seen` set.

## Quick start

1) Build the forked toolchain:

```bash
cd src
./make.bash
```

2) Run a fuzz target with LibAFL (example in this repo):

```bash
cd test/gosentry/examples/reverse
CGO_ENABLED=1 ../../../../bin/go test -fuzz=FuzzReverse --focus-on-new-code=false --catch-races=false --catch-leaks=false
```

Fuzzing runs until you stop it (Ctrl+C). The run prints `ok ...` on clean shutdown.

## Requirements

- `CGO_ENABLED=1` (required; the harness exports C ABI symbols)
- Rust toolchain + `cargo` (because the runner is `golibafl`)
- Repo layout assumption: `golibafl/` must live inside the active `GOROOT` directory (at `$GOROOT/golibafl`).
  - In this repo that means `.../gosentry/golibafl`.

## What happens under the hood

When `go test` sees `-fuzz` (and `-use-libafl` is true, which is the default), it does **not** execute Go’s native fuzzing coordinator/worker engine.
Instead it builds a **libFuzzer-compatible harness** and runs the Rust LibAFL runner against it.

### 1) `go test` builds a libFuzzer harness archive (`libharness.a`)

For the test package’s generated main package (`_testmain.go`), gosentry also generates an extra file (`_libaflmain.go`) and switches the link mode:

- buildmode: `c-archive` → produces a static archive: `libharness.a`
- exports:
  - `LLVMFuzzerInitialize`
  - `LLVMFuzzerTestOneInput`

Those are the standard libFuzzer entrypoints expected by `libafl_targets::libfuzzer`.

`LLVMFuzzerInitialize` selects the fuzz target that matches the `-fuzz` regexp and initializes the captured Go fuzz callback.

`LLVMFuzzerTestOneInput` converts the input bytes and calls the fuzz callback once.

Note: the exported entrypoints wait for Go package initialization to complete before running. This avoids spurious startup race reports when the harness is driven by an external runtime (LibAFL) that may call into the harness from multiple threads.

### 2) The fuzz callback is captured from standard `testing.F.Fuzz`

The `testing` package is patched so that, in this special mode, the first `f.Fuzz(func(*testing.T, []byte){...})` call is **captured** instead of starting the native Go fuzzing engine.

Then `testing.LibAFLFuzzOneInput` runs the captured callback in a normal `testing.T` context (using `tRunner`) so that `t.Fatal` / `t.FailNow` behave correctly (they call `runtime.Goexit`).

### 3) `go test` runs `golibafl` (Rust) with `HARNESS_LIB=...`

After building `libharness.a`, `go test` launches:

```bash
cargo run --release -- fuzz -i <input_dir> -o <output_dir>
```

with:

- `HARNESS_LIB=<path to built harness archive (ex: .../libharness.a or .../libharness_race.a)>`
- `HARNESS_LINK_SEARCH=/path/one:/path/two` (optional: extra native link search dirs)
- `HARNESS_LINK_LIBS=dylib=dl,static=z` (optional: extra `rustc-link-lib` entries)
- working directory set to the `golibafl/` crate

`golibafl` was updated to accept `HARNESS_LIB` and **link the prebuilt archive** instead of rebuilding a harness from a `HARNESS=...` directory.

### 4) Output directories

gosentry reuses Go’s fuzz cache root (roughly `$(go env GOCACHE)/fuzz`) so `go clean -fuzzcache` works.

LibAFL output is separated per **project + package + fuzz target**:

```
.../fuzz/<pkg import path>/libafl/<project>/<harness>/
  input/     # initial corpus dir (merged from testdata/fuzz + f.Add); may be empty
  queue/     # evolving corpus
  crashes/   # crashes (if any)
```

- `<project>` is derived from the package root directory (module root / GOPATH / GOROOT root) and formatted as `<basename>-<hash>`.
- `<harness>` is the fuzz target name when `-fuzz` is a simple identifier like `FuzzXxx` (or `^FuzzXxx$`), otherwise `pattern-<hash>`.

On each run, gosentry prepares `<...>/input/` as the initial `-i` corpus directory:

- files from `testdata/fuzz/` (if it exists) are copied into it
- manual seeds provided via `f.Add(...)` are written into it automatically
- if this fuzzing campaign was already run before, the previous LibAFL `queue/` corpus is automatically reused on restart (so Ctrl-C + rerun continues from the same corpus by default)

If the chosen `-i` directory is empty, `golibafl` generates a small random initial corpus.

On shutdown, `go test` prints the full output directory path:

```
libafl output dir: /full/path/to/.../libafl/<project>/<harness>
```

### Crash handling (stop on first crash)

In `--use-libafl` mode, gosentry follows `go test -fuzz` semantics: **stop the whole fuzzing run on the first crash** (even with multiple LibAFL clients).

When a crash is found, `golibafl` prints:
- the output directory + `crashes/` path
- the exact crash input file path
- a repro command: `golibafl run --input <crash_file>`

The harness output is printed above that (panic backtrace, or a stack trace for `t.Fatal`/`t.Fatalf`).

Example:

```text
Found 1 crashing input(s).

libafl output dir: /full/path/to/.../libafl/<project>/<harness>
crashes dir:      /full/path/to/.../libafl/<project>/<harness>/crashes

crash input:      /full/path/to/.../crashes/<id>
repro:            golibafl run --input /full/path/to/.../crashes/<id>

(Crash output is printed above; rerun the repro command to see it again.)
```

To keep fuzzing after crashes, set `"stop_all_fuzzers_on_panic": false` in the LibAFL JSONC config.
The crashing client is restarted after target crashes, including `SIGSEGV`.

Note: reproducing may require the same runtime environment variables as fuzzing (e.g. `LD_LIBRARY_PATH` for native deps).

## Current limitations

- Supports **multiple parameters** and gosentry’s fuzzable types:
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
