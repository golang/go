# Acceptance repairs for source 33752c03

These four slices repair failures observed during the preceding hosted
acceptance round. Their complete combined source tree is
`33752c03819ed5afcb37f87633a3cac668e181a8`. Each slice has one implementation
candidate and preserves its original frozen attacks.

| Slice | Contract and attack | Immutable component packet |
|---|---|---|
| M0-A-S20 / AUD-210 | Selected-package duplicate-stem rejection with complete module census, #442 | [Alias census](alias/README.md) |
| M0-A-S21 / AUD-211 | Public type-checker projection and private void-panic tracking, #443 | [Type projection](projection/README.md) |
| M22-A-S7 / AUD-212 | Bootstrap environment isolation and matched C/C++ toolchain, #444 | [Release environment](environment/README.md) |
| M22-A-S8 / AUD-213 | Shared parity admission before each child deadline, #445 | [Parity execution](parity/README.md) |

The packets retain the exact component commands, source/test hashes, RED and
GREEN records, caught mutations, restoration, and diagnostic limitations.
Original paths and pending statements remain historical. Later evidence
supplements those records without rewriting them.

## Combined installed acceptance

The [installed acceptance packet](../installed-acceptance-33752c03/README.md)
preserves the normal fresh build and all nineteen completed installed gate
records. The successful build took 480.027 seconds with final PGO and normal
dist cleanup. A preceding interrupted build has no completion receipt; its
cause is unconfirmed and its partial record remains available.

All nineteen commands matched their expected outcomes. Eighteen exited 0;
the intentional void-hook mutation exited 1 after exactly two builtin-void
attacks failed and seven controls passed. Restored selection passed all nine
cells with the same assertion binary. The full short type-checker suites and
short parity suite retain their documented expected skips. No release stamp
is present in these local records.

Execution used the frozen combined uncommitted snapshot based on local HEAD
`2535486fe9596a9212d525a5007963ef9e8682f0`, with index tree
`098de80a4ddf1e59b2750c1306ef495414f42d1e`. The source tree and twelve repaired
files stayed unchanged throughout the fresh build, gates, and subsequent
slice commits. This evidence does not assert a rebuild of each intermediate
commit or acceptance of a clean published release tip.

## Remaining requirements

The current [candidate record](../stabilization-candidate.md) and
[active request](../release-acceptance-request.json) identify the new source.
The full Go regression, all six no-short optimization cells and separate
`os` legs, actual SWIG/LTO coverage, unfiltered stamped release parity,
fresh hosted focused/runtime checks, and independent-host reproducibility
remain separate acceptance gates. The seven blocked attack rows are
unchanged. M6-S3b remains rolled back pending the repository's required
exception for an additional implementation attempt.

Every component packet has its own checksum list and manifest. The installed
packet has a separate checksum list and manifest binding its later evidence.
No packet contains build caches, installed tools, or diagnostic symlinks.
