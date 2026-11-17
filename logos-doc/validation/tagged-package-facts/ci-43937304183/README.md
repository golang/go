# Installed tagged package-fact proof

[Job 133545168161](https://github.com/gosuda/LoGOs/actions/runs/43937304183/job/133545168161)
passed on head `298f8969484af371bb58794610376dae77710a2b`, source tree
`8117b1f085a68753f359402393b7467a58eedacf`.

The archived old-production overlay returned exit 1 with exactly the required
two named failures and the assertion-metadata diagnostic. The restored
candidate produced all 150 required unique run/pass pairs: 142 leaf cases and
eight nonleaf roots. That includes the actual SIMD archive producer/import
test. No test skipped or failed; both commands had empty stderr and no timeout.

The independent audit counted the original event streams after downloading
the artifact. All six frozen hashes match. Source and installed tool snapshots
match before and after. No local Go execution was used to collect this proof.

- [Original artifact ZIP](tagged-facts-43937304183-1.zip), SHA-256
  `7d6b390e850ff0b778a9f4a80fa3d3e1006736c301d6229678d8ca5e9ec9b8dd`.
  It retains `negative.jsonl`, `candidate.jsonl`, before/after identities, and
  the complete acceptance report.
- [Raw job log](job.raw.log), [job metadata](jobs.json), and
  [artifact metadata](artifacts.json).
- [Independent audit](independent-audit.json) and [SHA-256 manifest](SHA256SUMS).

The deliberate old-code failures are required evidence. This successful gate
covers tagged package declarations; it does not validate the later method-key
codec or linker inline-context repairs.
