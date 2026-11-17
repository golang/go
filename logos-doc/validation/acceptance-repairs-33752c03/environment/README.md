# Release acceptance environment repair

Prepared 2026-11-16 for **M22-A-S7 / #444 / AUD-212**. This package records
one executor candidate against local base
`2535486fe9596a9212d525a5007963ef9e8682f0`. The compiler source at the original
failure and local native checks is
`e237c399aae8f8065b724e8a0e27585686c7da18`.

The candidate scopes resource limits to make, selects GCC 14 for both C and
C++, and places GOTMPDIR inside TMPDIR. It changes no compiler behavior,
upstream test, fixed exception, or timeout. The original executor and the
candidate are both included in `inputs/`; the frozen attack, coverage checker,
workflow, upstream tests, SWIG inputs, and 31-name exception script are under
`frozen/`. The 16 frozen inputs other than the executor remain byte-identical.

## Evidence and scope

| Gate | Actual result | Evidence |
|---|---|---|
| Existing bootstrap boundary | RED: six expected assertions across the four leaked variables, compiler mismatch, and temporary-root boundary; bounded-make/new-cache control passed | `runs/release-environment-red-001/` |
| Executor candidate | All four test methods passed | `runs/release-environment-green-001/` |
| Three independent reversions | Limits, C++ selection, and temporary-root mutations were each caught; restored candidate passed | `runs/release-env-*-mutation-001/`, `runs/release-environment-restored-001/` |
| Unchanged native runtime controls | Both memory-limit tests passed without skips, 4.258 seconds | `runs/environment-runtime-execute-002/` |
| Unchanged native security tests | Seven families passed without skips, 6.388 seconds | `runs/environment-linker-security-001/` |
| Existing coverage checker, Bash syntax, ShellCheck, actionlint | All four recorded commands passed | `runs/release-{static-checker,bash-syntax,shellcheck,actionlint}-001/` |
| Real SWIG ordinary/LTO paths | **Pending hosted execution with no skips**; SWIG is unavailable locally | Future installed/hosted gate |

The Python attack executes the actual extracted `bootstrap` function with a
restricted PATH. Download, build, and C-compilation calls are inert. This proves
the environment boundary and preserved make/cache safeguards; it does not
pretend to run a compiler, C23 program, SWIG, or the release workflow. Exact
commands, bound inputs, outputs, and results accompany every root-run gate.
The three mutated executors are included in `mutations/`.

The native gates used the already installed e96 toolchain and unchanged tests,
separately from that shell harness. The runtime test binary was compiled under
the original bounded build settings, then run with those settings cleared and
the corrected temporary roots and paired CC/CXX. Its first execution used the
wrong working directory and failed before child compilation. That setup-only
attempt and `runs/environment-runtime-setup-correction.json` are preserved;
they are neither a semantic RED nor another implementation attempt. The
identical binary and flags then passed from `src/runtime`. No compiled binary
is included in this portable package.

The five hosted security failures were `TestCheckGCCSpecFiles`,
`TestCheckLinkerWlFlags`, `TestCheckLinkerInputFiles`, `TestCheckLinkerCOFFDrectve`,
and `TestCheckLinkerObjectSections`. The seven-family native gate also includes
`TestCheckLinkerFlags` and `TestCheckLinkerSysrootFlags` as controls. The early
diagnosis memo mislabeled the first failed family as the sysroot control;
`diagnosis/errata.json` corrects that label without changing the historical
memo or any test. The additional GCC-spec excerpt retains the exact hosted
lines. The identity record proves all six inspected runtime/security/testing/
SWIG source files match pinned upstream Go 1.27.4 byte-for-byte.

## What remains pending

This is not a complete release acceptance result. Real SWIG ordinary and LTO
paths still require hosted execution with matched GCC 14 compilers and no
skips. The fresh combined build and installed gates remain separate. All six
complete optimization cells and their serial `os` stages must still pass;
the independent cmd/go shared-deadline failures, including the O3 failures,
are not resolved or waived by this package. No current e6e3 reproducibility
result is mixed into this environment repair proof.

The baseline hosted source is linked in `hosted/excerpt-provenance.json`:
[plain-Go O1 job](https://github.com/gosuda/LoGOs/actions/runs/43939662630/job/133553485876).
Raw excerpts keep original timestamps and line ranges. `copies.json` records
the source and SHA-256 of each byte-verified copy; `files.sha256` covers every
other regular file in the package. Historical absolute paths inside raw
receipts identify the actual executions and were not rewritten for portability.
