---
name: logos-performance
description: Route marked C calls through the fast-cgo entry in code built by logos. Use only when the user names fast-cgo, //logos:cgo_fastcall, #cgo noescape with #cgo nocallback, or GOEXPERIMENT=logosfastcgo; do not use for ordinary .lgo work such as ? error paths, unions, options, properties, or const views.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH.
metadata:
  version: "0.20.0"
  milestone: "M20"
---

# Fast cgo calls

The `logos` toolchain can route marked C calls through a fast entry instead of `runtime.cgocall`. The fast path applies only to marked functions in cgo packages built by `logos`, and only under the experiment flag. Everything else stays on the plain path.

## Procedure

1. Confirm the toolchain: run `logos version`. Done when it prints `logos version go1.27`. If the command is missing, stop and tell the user to build github.com/gosuda/loGos (`cd src && ./make.bash`) and put its `bin` directory on PATH.
2. Mark the C function with the full directive pair from the table below. Done when the function carries both halves and no half-marked function is expected to go fast.
3. Measure under the experiment: build and run with `GOEXPERIMENT=logosfastcgo`, then again without it. Done when the fast run is no slower than the plain run (gate: fast median ready-delay at most 1.05x the plain median; recorded fast/plain median ratio 0.71).
4. Ship the marking and keep the experiment flag in the build setup. Unflagged builds use the plain path by design, not by accident.

## Fast-CGo (M20)

Proven by `src/cmd/cgo/internal/testlogos/fastcall_test.go` (run with and without `GOEXPERIMENT=logosfastcgo`), `src/cmd/cgo/internal/testlogos/testdata/logosfast/c_fast.lgo`; the default-off state is pinned at `src/internal/buildcfg/exp.go:89`.

| cgo source | Fast entry |
|---|---|
| `#cgo noescape F` plus `#cgo nocallback F` in Go | the wrapper calls the fast entry |
| `//logos:cgo_fastcall F` in a `.lgo` cgo file | the same pair; the wrapper calls the fast entry |

Rules:

- The fast path needs the full pair: a C function with only `noescape` or only `nocallback` stays on the plain `runtime.cgocall` path (`TestFastcallEmission`). `GOEXPERIMENT=logosfastcgo` selects the fast entry; without it every call uses the plain path. The experiment is off by default.
- A fastcall name with no matched C function fails the build (`TestFastcallUnmatched`).
- A `nocallback` C function must never call back into Go: the violation panics at run time instead of hanging (`TestFastcallNoCallbackViolation`); the plain path follows the same protocol (`TestFastcallCallbackParity`).
- Blocking parity: a blocking fast call delays a ready goroutine no longer than the same sleep on the plain path. The gate fails when the fast median exceeds 1.05x the plain median (`TestFastcallBlockingParity`); the recorded fast/plain median ratio is 0.71.
- Profile samples during fast calls attribute to the Go call site, as on the plain path (`TestFastcallSigprofSurvives`, `TestFastcallProfileAttribution`).
- Applies to Go and `.lgo` cgo packages built by `logos` (`TestFastcallEmission` for the Go pair, `TestFastcallLogosSpelling` and `TestFastcallLogosEndToEnd` for the `.lgo` spelling). Builds without the experiment flag use the plain path.
