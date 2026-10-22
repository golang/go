# Lint ruleset for Go and loGo/s

Status: approved design. This document is the normative ruleset
specification for the lint system covering both plain Go packages and
loGo/s packages. Sources: the gVisor nogo discipline, the golangci-lint
catalog, and this repository's lint machinery. The engineering backlog
for this design lives in the implementation plan.

## Verdict

One engine owns depth. One policy file owns severity. One external tool
covers what the compiler cannot see.

- The in-compiler lint (cmd/compile/internal/logos/lint) stays the
  single deep engine for both languages. It sees lowering, escape
  facts, and ownership that no external tool can recover.
- The logos.toml [lints] table stays the single severity policy, and
  gains path classes with severity caps.
- logos-docs/golangci.yml covers repository Go tooling code only.

Rejected: running golangci-lint or cmd/vet over lowered .lgo output.
gVisor suppresses style analysis on generated code because it is
generated. Lowered .lgo output is generated code. An external tool over
that output would suppress everything that matters or flood false
positions. The two SSA-dependent analyzers worth having ride the
position-mapped vet path in phase 2 instead.

## Tier 1: correctness analyzers for both languages

The nine existing correctness and concurrency rules stay. Add the
gVisor-proven vet checks that the checker objects already carry.

Phase 1, type and syntax only. Each new rule registers in the existing
registry with an .lgo attack fixture and a _go.go twin.

| Rule | Group | Checks |
|---|---|---|
| correctness::printf_misuse | correctness | verb and argument count per call, including $"..." interpolation |
| correctness::unreachable_code | correctness | statements after return, panic, ? propagation, or terminal yield |
| correctness::lostcancel | correctness | a WithCancel or WithTimeout result whose cancel is dropped |
| correctness::copy_lock | correctness | a value containing a lock type copied by assignment or argument |
| correctness::bools_simplify | style, warn | always-true or always-false boolean expressions |
| correctness::string_intconv | correctness | string(int) and int(string) conversions |
| correctness::structtag_malformed | correctness | malformed json or logos struct tags |
| correctness::stdmethods_shape | correctness | wrong signatures for Read, Write, MarshalJSON, and the binary family |
| correctness::tests_malformed | correctness, Go only | malformed Test or Benchmark signatures in _go.go twins |

Phase 2, SSA-dependent, blocked on the vet-over-lowered path:
nilness joins then. Shadow stays off everywhere. gVisor disabled it
with a stated reason, and this fork's inliner work increases its noise.

## Tier 2: deep loGo/s rules

These rules consume compiler facts, so no external tool can have them.

- perf::zeroalloc_violation stays unconfigurable and unsilenceable. The
  annotation is a proof obligation, not a preference.
- concurrency::concurrent_multiword_race and
  concurrency::reentrant_withlock_deadlock stay Forbid.
- correctness::arena_pointer_escape stays.
- perf::large_backing_array_pin stays escape-phase resolved.
- One new rule, the checklinkname analog:
  portability::linkname_discipline, severity Deny. It flags a
  go:linkname outside logos/rt and runtime files that names a runtime
  symbol not on the allowlist beside the rule, each entry with a
  reason.

## Tier 3: slop, complexity, suspicious, style

Existing rules stay. Two additions:

- slop::empty_critical_section: a WithLock callback or lock pair whose
  body touches no protected state.
- pedantic::unused_allow: a logos:allow(rule) that matched no finding
  in the file. This is nolintlint strictness on the fork's own
  mechanism, and it keeps inline allows from rotting.

## Policy model: path classes

The logos.toml [lints] table gains a scope block. Classes cap what can
fire per path; Forbid correctness stays on everywhere.

```toml
[lints.scope]
generated = "correctness+concurrency+perf"
external  = "correctness"
internal  = "all"
```

Mapping: generated matches emitted or vendored paths (files named with
the _logos_ prefix, src/vendor, cmd/vendor). External matches
dependency modules outside the main module. Internal is everything
else. Severity caps, not rule lists: one question, one answer.

Rejected: per-rule per-path tables in the raw nogo.yaml style. The
group ladder plus path classes answers the same question in one layer.

## Suppression discipline

- logos:allow(rule) above the statement stays the only inline
  suppression. The package-clause allow for alias_extension stays.
- An allow must name a rule. An unused allow is itself a finding.
- zeroalloc accepts no allow.
- Centralized carve-outs live in one file beside the registry, each
  entry with a reason. Repository code gets none until a review names
  one.

## golangci.yml for repository Go tooling

From default: standard to a curated set. Scope stays new packages
only, fork go first on PATH.

Keep: errcheck, govet, ineffassign, staticcheck, unused.

Add: errorlint with asserts off, unconvert, unparam, noctx, intrange,
misspell with US locale, copyloopvar, gocheckcompilerdirectives,
gochecknoinits with the one registration-path carve-out, nolintlint
strict, and govet nilness.

Leave off, reasons recorded in the file: shadow, gocritic tags beyond
diagnostic, dupl, funlen, lll, QF1001, QF1008.

Add one depguard rule, the fork's own boundary: logos/rt and logos/sync
may not import each other, and nothing under cmd/compile imports
logos/rt outside the lowering seam files.

## What this ruleset deliberately does not do

- No lint rule runs only under logos lint while builds ignore it,
  outside the documented exceptions.
- No style rule is Forbid. Style tops out at Deny. Only correctness
  and concurrency forbid.
- No new thresholds. The six existing thresholds cover the complexity
  family.
- No staticcheck family import into the compiler. A staticcheck check
  becomes a native rule only when a red-team row demands it.

## Verification

- Every new rule lands with its attack fixture and _go.go twin where
  language neutral. TestRuleScopes extends. The registry-count
  assertion updates.
- BenchmarkLintOverhead stays under the 2 percent budget with the full
  phase-1 set on. A rule that breaks the budget does not land.
- logos_lint_config.txt gains one leg per scope class.
- The golangci.yml upgrade is proven by one run over the covered
  packages before any new package depends on it.
- Phase-2 analyzers land only after the vet path exists, with a
  fixture proving a finding in a .lgo file reports the .lgo line.
