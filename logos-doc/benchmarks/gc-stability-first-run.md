# First native GC stability run

Date: 2026-11-16

Source head: `0b56e0c353d0d57b7b1f2f88ac073db535986d88`.

[PR run 43825911296](https://github.com/gosuda/LoGOs/actions/runs/43825911296) tested merge commit `0f6cc74f`. Native linux/arm64 builds with `logosarena` and `logosarena,nogreenteagc` both passed toolchain build, gofmt, and vet, then failed the runtime regression step. Runtime metrics, pprof, and trace packages passed. Subsequent race, repetition, and checkmark steps were skipped after the runtime failure. A separate push run reproduced the same failures. No amd64 or upstream milestone result is claimed here.

## Reproduced ownership defect

The unchanged `TestLogosRegionReleaseOwnership` reported a published explicit-region root also present in a reusable P pool in all three completed runs:

| Collector and event | Job | Published pooled head |
| --- | --- | --- |
| Green Tea, PR | [133159711773](https://github.com/gosuda/LoGOs/actions/runs/43825911296/job/133159711773) | `0x6811e286c000` |
| Non-Green Tea, PR | [133159711427](https://github.com/gosuda/LoGOs/actions/runs/43825911296/job/133159711427) | `0x54442bd6e000` |
| Green Tea, push | [133159591943](https://github.com/gosuda/LoGOs/actions/runs/43825904706/job/133159591943) | `0x54c8c31a2000` |

These are ordinary runtime builds. A no-inlining configuration was not needed to observe the invariant violation. The failure establishes overlapping published and reusable ownership, not a demonstrated use-after-free crash.

Keep the ownership helper and behavioral test unchanged while repairing the release order. A later mutation probe must restore the original production file and reproduce this assertion.

## Fixture corrections

The first run also found two independent test problems. They are not runtime defect REDs.

- The Interface layout oracle counted both words as traced pointers. The compiler's documented type-bit rule in `src/cmd/compile/internal/typebits/typebits.go` traces only the data word; type metadata is retained separately. Correct the declared oracle offset to one pointer word. The byte-by-byte sweep and expected extent calculation remain unchanged.
- Six older compiler-seam tests assumed arena `make` lowering on every architecture. `walkLogosArenaMake` explicitly returns without lowering outside amd64. Apply the existing `TestLogosArenaMakeSeamPlacement` architecture guard to the dynamic-make watermark, panic-unwind, loop-watermark, and three overflow tests. Their amd64 assertions remain unchanged. Direct runtime allocator, region, scanner, ownership, and heap-graph tests remain enabled on arm64. A new direct `LogosArenaMakeSlice` test reuses the existing overflow table on every architecture, preserving runtime overflow and diagnostic coverage independently of compiler lowering.

On arm64, the unlowered compiler seam can use ordinary stack slice allocation while its heap control uses `makeslice`; overflow diagnostic ordering can differ between those paths. Changing runtime panic behavior to make this unexercised arena comparison pass would test the wrong mechanism.

The corrected fixtures and ownership repair require fresh execution. This record preserves the original failures rather than recasting fixture errors as successful attacks on the runtime.
