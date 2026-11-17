# Concurrent branch integration, 2026-11-16

The public stabilization branch advanced from `45c29ff2` to
`8afec95580ed24e593307fd9d30d626eff31e22d` while the separate acceptance
follow-ups were being checked. That head includes release
`7d15ec7671582d0f022013bf36cf8f82a1b04cfc`. This integration preserves the
concurrent commits as an ordinary merge parent.

`metadata.json` records exact parents and full/source trees. The two TSV
files preserve complete changed-path inventories. They were copied without
modification from the read-only review. No workflow changed in the concurrent
branch; its golangci configuration change is a comment correction.

## Source composition

- Preserve the incoming shortest-rooted capability graph and discovery-order
  tests, including the explicit five-second child timeout.
- Preserve native array-index diagnostics, their Go/loGo/s tests, and the
  active specification. Correct their stale references from #333 to #440.
- Preserve overlay-aware inactive ejection and both backing-source exclusion
  assertions. Compose its `eject.go` changes with the separately verified
  test-adapter relocation and cleanup error check.
- Preserve the concurrent sequential SIMD guard tests and baseline CPU
  settings. Native generated-SIMD execution stays unconditional.
- Preserve static quoted-text element-type inference. The following JSON
  commits separately correct the profile placement of the unchanged append
  regression and compose the reviewed shared-decoder implementation.

The CTFE oracle correction has no incoming fixture conflict. Compiler JSON
lowering files and their C-aware source printers match source45 exactly.

## Evidence and ledger reconciliation

The imported-symbol directory keeps both complete historical snapshots, as
described in `../../imported-local-types/MERGE-2026-11-16.md`. No historical
manifest is rewritten to accommodate a merged README.

The concurrent canonical IDs are retained: Close, panic identity, and imported
symbols use #333, #334, and #335; the GC attacks use #430, #431, and #432.
Earlier source-45 evidence keeps its original #416/#417/#418 and GC numbers.
Those historical references do not count as additional active defects.
Active rows are sorted numerically, and the historical GC paragraph now
identifies the renumbered row explicitly.

New tracking records cover capability discovery, array-index diagnostics,
and overlay ejection. The previously omitted M18-A-S20/S21/S22 task rows are
added to both identical task ledgers without changing their implementations
or acceptance status. Pending mutation and installed gates remain explicit.

Source-45 results remain valid at their recorded revision. They do not
establish acceptance of this combined source. A fresh build, all affected
installed gates, and the complete Go regression remain required.
