// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64 || amd64p32 || arm || arm64 || loong64 || mips64p32le || mipsle || mips64le || ppc64le || riscv || riscv64 || wasm

package rt

// BinaryHostLittleEndian reports whether an int64, uint64, float32, or
// float64 lies in memory as the little-endian bytes of the SPEC 10.3 binary
// wire format. Generated code copies slices and arrays of such elements in
// one call when it holds, and keeps the per-element loop otherwise.
const BinaryHostLittleEndian = true
