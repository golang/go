# Plugin test precondition correction

The original frozen script expected a plugin built by a plain-Go child to
load into a default `logos test` process. That expectation combined two
independent settings differences. The host enables `logosarena` and defaults
to O2; plain Go defaults to O1. This script was corrected under G1 before the
environment repair was reapplied.

The first environment-only implementation repaired the test and subprocess
`GOEXPERIMENT` mismatch, but the unchanged plugin assertion still failed on
`math/bits`. Three probes used the same frozen Go test and plugin source:

| Host and inherited settings | Result |
| --- | --- |
| Repaired `logos test`, inherited `GOFLAGS=-O=2` | Plugin loads and returns 42 |
| Repaired `logos test -O=1`, child uses plain Go's default O1 | Plugin loads and returns 42 |
| Plain `go test -O=2`, matching explicit arena/architecture environment, child uses default O1 | Plugin rejects `math/bits` identity |

This isolates the remaining default-profile mismatch to the documented
optimization settings. `src/plugin/plugin.go` requires matching toolchain,
build tags, and relevant flags/environment values for the host and plugins.
No automatic forwarding of all compiler flags is proposed.

The production repair was rolled back before this test-only commit. The
corrected script supplies `-O=1` to its logos host, matching the child driver's
ordinary default. It still requires actual plugin compilation, loading,
symbol type identity, and the value 42, with no changed assertions. The
default arena case must still fail against the unchanged driver because its
child loses `GOEXPERIMENT`; that RED is recorded before reapplying the repair.

The original script bytes, frozen hashes, failure logs, and complete probe
commands are retained here. The original upstream xxhash corpus is unchanged.
Its unconfigured O2/O1 plugin cell is not a successful comparison and must
remain separate from any explicitly matched O1 or O2 replay.
