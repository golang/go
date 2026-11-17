# Existing const-provenance acceptance

On 2026-11-16 the frozen45 installed compiler passed eight existing M5
compiler fixtures and three selected cmd/go scripts under a reusable race
test binary, with no skips. The alias script rejected all ten consumers
through both native build and Go ejection, then passed its native and
upstream-ejected accepted controls.

These results resolve the specific stale M5 planning notes about existing
const-origin checking and the completed separation of checked-source
rejections from native placement probes. They do not certify every unsafe
flow, every platform, or a complete language release.

See [the dated execution record](EXECUTED-2026-11-16.md), `summary.json`,
and the immutable command inputs and raw logs in `testdir/` and
`scripts-race/`. The original thirteen-file pre-run manifest remains
unchanged; `observed-controls.sha256` separately identifies the existing
positive address fixture also selected by the gate.

No M5 implementation, fixture, normative specification, or assertion was
changed. No new mutation probe was run. `evidence.sha256` identifies the
copied text records; test binaries, caches, and live source temps are not
included.
