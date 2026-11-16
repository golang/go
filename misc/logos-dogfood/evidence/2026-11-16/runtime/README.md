# Runtime validation of candidate three

The completed checks comprise **14 successful package-test invocations** and
**two successful program executions**. Every package-test command used
`-count=1`. Ejection and lint analysis are separate operations and are not
included in those execution counts.

The modernc test has **no completed `logos` execution**. Its unconfigured run
was denied by the capability loader. A separately configured run passed that
stage but its compiler process was killed while compiling the generated SQLite
library. Neither run reached a test function. The latter result is incomplete
runtime evidence, not proof of a compiler correctness defect.

## Verified toolchain

These runs used the compiler built from production revision
`4cbfbf5af30d99609d385afbecfa305026a32307`. Later commits changed tests,
documentation, or the corpus runner without changing these installed binaries.
Each report records its actual checkout revision, compiler source digest,
driver and tool hashes, source digest, command, environment overrides, duration,
exit code, and log hashes.

| Input | SHA-256 |
| --- | --- |
| Stock Go 1.27.4 driver | `de0ae3e082ce7599eae4fb69fa2ef1e43b8f31ffd3fce62cdf81930aa9005050` |
| Fork `go` and `logos` drivers | `4c016af1e3b0b2df74f0056a878b994cb6194887273982cd9f09606e0923e3a5` |
| Fork compiler | `7c519b46141178df983fe62f9bbb7ef794201dba8daf31a676143ed657373f37` |

The compiler source and binary manifests, with the separate compiler/lint unit
gates, are in the [compiler validation report](../../../../../logos-doc/validation/dogfood-lint/README.md).
All runs used linux/amd64, one package worker, `GOMAXPROCS=1`,
`GOMEMLIMIT=512MiB`, disabled automatic downloads, and the existing shared build
caches. The fixture runner inherited the memory limit and temporary directory
from its parent environment; the focused replay records them explicitly.

## Executed checks

| Source and check | Stock Go | Fork Go | `logos` | Successful package-test invocations |
| --- | --- | --- | --- | ---: |
| Semantic fixture, all seven tests | Pass | Pass | Pass | 3 |
| C ABI fixture, its boundary test | Pass | Pass | Pass | 3 |
| Lint fixture, two safe runtime controls | Pass | Pass | Pass | 3 |
| mattn SQLite `TestOpen`, `TestClose`, `TestInsert` | Pass | Pass | Pass | 3 |
| modernc SQLite `TestMutexWakeOnce` | Pass | Pass | Uncompleted | 2 |

The [fixture report](fixtures/report.json) uses the normal manifest `test`
operation, with no build override and no test-name filter. Its arithmetic test
checks every pair of signed 8-bit values against a wider reference; the other
tests cover slice capacity and panic order, overlapping copy and append,
assignment destinations, iterator cleanup and closure lifetime, generic named
integers, and cross-goroutine mutex release. The cgo fixture checks seven
unsigned boundary values, 49 C struct-argument combinations, and pointer writes.

The [SQLite report](sqlite/report.json) records the exact upstream pins:

- `github.com/mattn/go-sqlite3` at `4cddc794daa0340d50de741dae025421bbdbba0c`, with `CGO_ENABLED=1`.
- `modernc.org/sqlite` at `92d72a45bcb65ab33bbe79c8667f2b8b529ae902`, with `CGO_ENABLED=0` and its original `modernc.org/libc v1.80.1` dependency.

Every focused test run uses JSON events. The replay requires exactly one `run`
and one `pass` event for each named test and rejects skips, failures, missing
tests, or additional test names. No source or dependency was edited. Every
command operated on a fresh verified private source copy.

### Native and ejected language behavior

Native `logos run .` and stock Go execution of `logos eject` output both exited
zero and produced exactly `language semantics ok\n`. The fixture exercises
optional fallback evaluation, switch expressions, and successful and failing
try assignments, including preservation of the returned value on failure.
These are two program executions; the conversion itself is not a runtime test.
The [language report](language/report.json) records the original and ejected
tree digests and the exact output hashes.

### Exact lint diagnostics

The source-reviewed oracle was written before running lint. Two fresh-copy
executions matched the complete expected records, including severity, message,
source span, byte offsets, provenance, and absence of extra findings:

| Source location | Expected finding |
| --- | --- |
| `cases.go:34:2` | `correctness::unclosed_io_closer` for `Leak` |
| `cases.go:39:2` | `correctness::unclosed_io_closer` for the short-circuit close |
| `cases.go:54:6` | `correctness::infinite_recursion` after a returning shadowed `panic` |

The checked, assigned, and returned close forms produce no leak finding. The
parenthesized builtin panic does not produce a recursion finding. This fixture
is plain Go: its local pointer variable returned as `error` is outside the Go
typed-nil rule's explicit-conversion scope. A separate runtime control verifies
that this typed-nil interface is non-nil under all three drivers; it is not
claimed as a lint finding. See the [expected records](lint/expected.json) and
[lint/runtime report](lint/report.json).

## Uncompleted modernc `logos` execution

The original `logos test` attempt produced six unsafe-import denials and two
assembly denials before compilation. Its raw report still has the runner's
`candidate-failed` status. Source review identifies this outcome as the
documented capability requirement, and zero tests are counted for it.

The separate [configured report](modernc-policy/report.json) applied the exact
reviewed [policy bytes](modernc-policy/policy.toml), SHA-256
`5b2a173a8d779e0a92587042884eddaa3a5c30a3ae11a1b913beb53cac4f6c12`, only to a
private copy. The six package grants do not change severity defaults or replace
dependencies. The report retains both the pristine source digest and the policy
digest, and verifies its passing stock-Go baseline.

That attempt ended after 121.341 seconds with
`modernc.org/sqlite/lib: .../compile: signal: killed`. The enclosing 600-second
deadline did not expire. Host memory was shared with other validation campaigns;
the log alone does not establish the cause of the kill. The configured runtime
test remains pending a serial retry after those campaigns release resources.

## Reproduce

From the repository root, provide paths to the original stock Go toolchain,
the pinned Git source root, the prepared module cache, a reusable cache root,
and a new output root. Keep the built fork installed at `bin/go` and
`bin/logos`. For example, with those paths in `DOGFOOD_ORACLE`,
`DOGFOOD_SOURCES`, `DOGFOOD_MODULES`, `DOGFOOD_CACHE`, `DOGFOOD_TMP`, and
`DOGFOOD_OUT`:

```sh
GOMEMLIMIT=512MiB TMPDIR="$DOGFOOD_TMP" CC=/usr/bin/cc CXX=/usr/bin/c++ \
uv run --no-project python misc/logos-dogfood/run.py \
  --go-oracle "$DOGFOOD_ORACLE" --go-candidate "$PWD/bin/go" \
  --logos "$PWD/bin/logos" --source-root "$DOGFOOD_SOURCES" \
  --module-cache "$DOGFOOD_MODULES" --cache-root "$DOGFOOD_CACHE" \
  --only fixture-semantics --only fixture-cgo \
  --stage oracle --stage go --stage logos --jobs 1 --timeout 180 \
  --out "$DOGFOOD_OUT/fixtures"

for part in language lint sqlite; do
  uv run --no-project python misc/logos-dogfood/evidence/2026-11-16/runtime/replay.py \
    --oracle "$DOGFOOD_ORACLE" --candidate "$PWD/bin/go" \
    --logos "$PWD/bin/logos" --source-root "$DOGFOOD_SOURCES" \
    --module-cache "$DOGFOOD_MODULES" --cache-root "$DOGFOOD_CACHE" \
    --tmpdir "$DOGFOOD_TMP" --part "$part" --out "$DOGFOOD_OUT/$part"
done

uv run --no-project python misc/logos-dogfood/evidence/2026-11-16/runtime/policy_replay.py \
  --logos "$PWD/bin/logos" --source-root "$DOGFOOD_SOURCES" \
  --module-cache "$DOGFOOD_MODULES" --cache "$DOGFOOD_CACHE/logos" \
  --tmpdir "$DOGFOOD_TMP" --policy misc/logos-dogfood/policies/modernc-sqlite.toml \
  --policy-sha256 5b2a173a8d779e0a92587042884eddaa3a5c30a3ae11a1b913beb53cac4f6c12 \
  --baseline "$DOGFOOD_OUT/sqlite/report.json" --out "$DOGFOOD_OUT/modernc-policy"
```

The two replay scripts passed Ruff formatting/lint checks and `ty` typechecking.
Reports and logs here are unmodified copies from their original run directories;
absolute paths in the reports name those original locations. Log files are
mirrored under the same relative names. Generated ejected code and temporary
source copies are omitted; the replay regenerates them.
