### SIMD {#simd}

The new [simd] and [simd/archsimd] packages provide access to SIMD
(single instruction, multiple data) operations. The [simd] package
provides portable, vector-size-agnostic types and operations, and the
[simd/archsimd] package provides architecture-specific vector types and
operations on amd64, arm64, and WebAssembly.

These packages, and the compiler support for them, can be disabled by
setting `GOEXPERIMENT=nosimdstable` at build time. This setting is
expected to be removed in a future release.

Additional, experimental API, including arm64 SVE support, is available
by setting `GOEXPERIMENT=simd` at build time. That API is not subject
to the Go 1 compatibility promise.

TODO: Expand.
<!-- go.dev/issue/83009 -->
