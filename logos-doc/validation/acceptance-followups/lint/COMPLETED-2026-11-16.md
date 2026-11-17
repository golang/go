# Completed lint follow-up checks, 2026-11-16

The earlier [component record](README.md) and its checksum manifest remain
unchanged. They describe the boundary before the later bounded vet retry and
cleanup mutation. The [new execution summary](bounded-retry/summary.json)
records those completed checks against the same five source files.

| Check | Result |
|---|---|
| Normal eject vet | Exit 0, no diagnostics, 8.305 seconds |
| External cleanup-condition mutant | Exit 1, all four existing target cases caught the false cleanup error, 4.190 seconds |
| Fresh no-overlay targeted race binary | Exit 0, the same four cases and parent passed without skips, 5.158 seconds |

The mutation inverted only `err != nil` to `err == nil`. The source tree was
never mutated. The clean control was freshly compiled without an overlay, and
all five source hashes still match the original transformation proof.

Together with the preserved earlier checks, the configured full lint scope
has zero findings and all 100 existing eject race cells pass. The first vet
attempt remains an interrupted observation; it is not counted as a passing
gate. No test inputs or assertions were weakened.

These are component checks with the freshly installed source-45 compiler.
Fresh build, normal vet, and complete regression of the final merged source
remain separate acceptance gates. `completion.sha256` covers this completion
record and the new bounded-retry files. It does not replace the original
`checksums.sha256` or either historical frozen test manifest.
