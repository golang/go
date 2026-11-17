# M20 cgo candidate integration

This record describes the combined review stack. It does not claim that the
combined production gates or G6 acceptance have passed.

## Review order

1. [PR #29](https://github.com/gosuda/LoGOs/pull/29): ordinary-cgo result frames,
   shared experiment and validation infrastructure.
2. [PR #41](https://github.com/gosuda/LoGOs/pull/41): package-wide contracts,
   first production commit `0f8142de1ef63523a29df472720f549f7de6ba66`.
3. [PR #38](https://github.com/gosuda/LoGOs/pull/38): validated-span reuse,
   first production commit `3dc356d9adecb4593f58f3caed644c30ba26be68`.
4. [PR #37](https://github.com/gosuda/LoGOs/pull/37): pointer-free typed addresses,
   first production commit `31acd3d7e4a01309d60c4f79301d82357d8815b6`.

The feature merge commits keep both branch parents. No history is rewritten
and no default-branch merge is part of this stack.

## Preservation checks

The integration reconstructed each branch's diff against shared parent
`634d04cb2c8eb8696562721be2a4ae6228140f79` and checked that it reproduced the
exact published source before combining the changes. All sixteen test,
fixture, benchmark, and test-export source files remain byte-identical to
their respective first production commits. The runtime source keeps both
small hooks; the unchanged pointer bitmap loop and fallback paths are retained.

The manifest is an additive union: two runtime families with five required
top-level tests, three direct fixtures, four independent target benchmarks,
and one explicitly non-floor reference. All six mutation/counterfactual
probes remain present, and every source replacement matches exactly once.
The original per-slice manifest hashes remain historical evidence; the
combined manifest adds the other slices without changing any declaration.

## Why the targets remain separate

The two result-frame rows use scalar arguments. The span row passes a saved
unsafe pointer with a nil marker, which cannot enter the typed-address
shortcut. The typed-address row's baseline returns for a pointer-free type
before reaching the unknown-object scanner. Both runtime rows call void C
functions, so result-frame specialization cannot explain their timing.

The package-contract probe inspects generated C text. Runtime shortcuts
cannot hide its missing-elision failure. The span zero-base probe uses
unknown pointers with nil markers. The typed-address inversion probe skips
a required check before the span path could run. Each faulty predicate must
therefore still be rejected with the other optimizations present.

## Evidence already completed

At the shared parent, [run 43831729270, job 133178423012](https://github.com/gosuda/LoGOs/actions/runs/43831729270/job/133178423012)
passed the build, formatting and explicit off/on vet, both existing
return-frame mutation/restoration probes, all off/on normal/race/cgocheck2
focused and pointer-check suites, all four direct ASan/MSan fixture runs,
and the stock-Go fixture. It completed at 08:52:31 UTC on 2026-11-16.
The strict callback stack-movement assertion passed after the scoped
test-observer correction.

That parent result is not a pass for the new runtime or contract-collection
implementations. Their separate test-first RED records are in M20-S3,
M20-S4, and M20-S5. Combined GREEN, new probes, complete default-short dist
regression, and the frozen benchmark campaign remain required.

## Performance acceptance remains separate

The benchmark runner preserves per-row baseline, candidate, and fresh
integration measurements, noise checks, fixed work, and separate profiles.
A latency pass is not G6 acceptance. The current native-call and typed-pointer
references do not establish an exact same-contract lower bound; the
baseline/floor admission ratio remains unresolved. Cumulative cgocall profile
time includes descendant work and must not be relabeled as irreducible cost.
Keep `logoscgoopt` off by default and report the unresolved gate explicitly.
