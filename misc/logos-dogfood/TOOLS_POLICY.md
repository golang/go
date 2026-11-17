# The tools build permission profile

`policies/tools-build.toml` is an explicit compile/link profile for the pinned
`golang.org/x/tools@v0.52.0` input. It preserves the original three unsafe grants,
adds process permission for two exact packages, and adds network permission for
14 exact packages. It changes no severity default and grants no wildcard or
errgroup permission. The original `policies/tools.toml` and its
`corpus-permissions.json` row remain the strict diagnostic profile.

## Evidence and limit

[Hosted run 43950476822](https://github.com/gosuda/LoGOs/actions/runs/43950476822)
compared all three drivers at head
`7d5446cca8e5b65800732c27bee2354ce8ad1461`, native source tree
`df267333b1d5c5727e2477215a6807cae52f8e18`. Stock Go and the fork Go driver built
the input successfully. LoGOs exited 1 after 35.218 seconds without a timeout.
Its strict probe artifact, `tools-probe-43950476822-1`, has ZIP SHA-256
`f28b83d3d7b7a57a1fa84f945bb0a0aaf451841bc596f1aeac5d935e019cc89b`.

The module remains revision `16517d951552611e7bd358c48ee1051fd0e57b0c`, with sum
`h1:ryxg4U63oaUYk7uOLTF37n43f4qgtGvqV4Ioit1BaEs=`. The original policy remains
SHA-256 `005ba0707611025ef364004190601f0f103ab28ab97da9a6acd81e034d7f4d3f`.
The probe records the native driver, compiler, linker, source revision, and
compiler-source hashes. All 20 recorded output hashes and the saved policy
bytes were checked against the downloaded artifact. Source review used module
trees rehashed against the pinned module sums.

The fresh capture has 42 capability errors: 36 network and six process errors.
The old capture had 44. The four previously reported errgroup network errors
are absent from the fresh capture, and two additional diagnostics are visible.
Both captures hit the linker's error limit. In the fresh capture, cmd/deadcode
and cmd/present each stop after 21 errors. These counts do not establish an
exhaustive permission set. An unlimited strict diagnostic pass must retain
`-ldflags=-e`, the original policy, and the same source/tool/resource identities.
Any newly exposed package needs its own source and path review.

This probe builds packages. It does not execute their tests or the reported call
paths. The full corpus acceptance and runtime comparisons remain separate gates.

## Why the exact package grants are present

[SPEC 11.1](../../logos-doc/SPEC.md#111-sink-symbols) requires synthetic edges for
interface calls and reflective method lookup. The residual captured network
paths use those conservative edges. They do not prove that the concrete values
in this application perform network operations. This profile accepts that
analysis contract and keeps its permissions explicit.

| Package | Source and captured path |
| --- | --- |
| `golang.org/x/telemetry` | `start.go:178` starts a child with `exec.Command` and `Cmd.Start`, supporting process permission. Its captured network path follows Start's cleanup through `io.Closer.Close`. |
| `golang.org/x/telemetry/internal/configstore` | `download.go:46` runs `go mod download`, supporting process permission. Its network path follows Cmd.Wait's `io.Closer` cleanup. |
| `golang.org/x/telemetry/internal/counter` | `counter.go:26` writes debug output with Fprintf; conservative `io.Writer` edges reach TLS and connection cleanup. |
| `golang.org/x/telemetry/internal/crashmonitor` | `monitor.go:36` reads stdin through ReadAll; conservative `io.Reader` edges reach TLS. |
| `golang.org/x/telemetry/internal/mmap` | `mmap_unix.go:16` calls File.Stat; error and String method edges reach Fprintf and writer implementations. |
| `golang.org/x/telemetry/internal/telemetry` | `dir.go:124` calls ReadFile; its file-error path reaches the same method/writer edges. |
| `golang.org/x/telemetry/internal/upload` | Initialization calls MustCompile, whose error/String/writer path reaches network sinks. Separately, `upload.go:93` directly calls http.Post. |
| `github.com/yuin/goldmark/ast` | `ast/block.go:473` calls Printf; Fprintf's writer edge reaches an HTTP chunk writer. |
| `github.com/yuin/goldmark/parser`, `github.com/yuin/goldmark/util` | Initialization calls MustCompile; error/String/writer edges reach HTTP. |
| `github.com/yuin/goldmark/renderer` | `renderer/renderer.go:135` flushes its writer; the writer edge reaches an HTTP chunk writer. |
| `github.com/yuin/goldmark/renderer/html` | `renderer/html/html.go:447` writes through a writer interface, reaching the same HTTP branch. |
| `github.com/yuin/goldmark/text` | `text/reader.go:528` calls FindReaderSubmatchIndex; RuneReader/Reader edges reach HTTP reader/writer implementations. |
| `golang.org/x/net/websocket` | `websocket.go:221` writes and closes a payload writer; its Close edge reaches an HTTP client connection. The package also owns websocket I/O and DialContext in `dial.go:13`. |

These sources are telemetry `v0.0.0-20261009151841-97e8ea77aa2a`, Goldmark
`v1.4.13`, and x/net `v0.61.0`, all pinned by the tools module graph. No changes
are made to those sources or to the capability auditor to enable this profile.

## Unlimited strict diagnostic capture

The workflow first retains the three-driver strict build from
`corpus-permissions.json`, then invokes `probe_tools.py` against that report:

```sh
uv run --no-project python misc/logos-dogfood/probe_tools.py \
  --logos "$GITHUB_WORKSPACE/bin/logos" \
  --source-root "$INPUTS/sources" --module-cache "$INPUTS/modules" \
  --cache-root "$RUNNER_TEMP/corpus-cache" \
  --baseline "$EVIDENCE/tools-probe/report.json" \
  --jobs 2 --timeout 600 --out "$EVIDENCE/tools-unlimited"
```

The helper makes its own verified private source copy, applies the original
strict policy, and runs the explicit command `logos build -ldflags=-e ./...`.
It binds the earlier successful stock and fork builds to the exact profile,
commands, working directory, resource environment, executable identities, and
raw output hashes. It records the native source tree and executable receipts
before and after the probe, then rechecks the source, policy, baseline, and
private effective-source digest. The existing runner supplies source copying,
policy validation, offline environment, process capture, and hashing unchanged.

`complete: true` means that the diagnostic capture completed with stable inputs.
A normal build exit of 1 is recorded as `build_succeeded: false`; it is not corpus
acceptance. Timeout, process failure, changed inputs, malformed baseline exit
codes, or another error-limit message fail the capture. The helper records the
actual inherited `GOGC` presence and value in addition to the runner's environment
receipt. The workflow uses the same resource settings for both captures.

Python controls exercise these boundaries without running Go. Native execution
of this new helper remains a hosted CI gate. The early artifact retains both
strict captures and their source/tool/resource receipts. The full 196-cell
replay and its existing acceptor remain independent and unchanged.
