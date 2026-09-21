// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests for scaled index addressing
// with non-power-of-two slice strides (issue #80639).

//go:noinline
func idx3(p []uint32, x int) uint32 {
	// arm64: `MOVWU\s*\(R[0-9]+\)\(R[0-9]+<<2\)`
	// amd64: `MOVL\s*\([A-Z0-9]+\)\([A-Z0-9]+\*4\)`
	return p[3*x]
}

//go:noinline
func idx7(p []uint32, x int) uint32 {
	// arm64: `MOVWU\s*\(R[0-9]+\)\(R[0-9]+<<2\)` -`SUB\s*R[0-9]+<<2`
	// amd64: `MOVL\s*\([A-Z0-9]+\)\([A-Z0-9]+\*4\)`
	return p[7*x]
}

//go:noinline
func idx11(p []uint32, x int) uint32 {
	// arm64: `MOVWU\s*\(R[0-9]+\)\(R[0-9]+<<2\)`
	// amd64: `MOVL\s*\([A-Z0-9]+\)\([A-Z0-9]+\*4\)`
	return p[11*x]
}

//go:noinline
func idx5_64(p []uint64, x int) uint64 {
	// arm64: `MOVD\s*\(R[0-9]+\)\(R[0-9]+<<3\)`
	// amd64: `MOVQ\s*\([A-Z0-9]+\)\([A-Z0-9]+\*8\)`
	return p[5*x]
}

//go:noinline
func idx7_16(p []uint16, x int) uint16 {
	// arm64: `MOVHU\s*\(R[0-9]+\)\(R[0-9]+<<1\)` -`SUB\s*R[0-9]+<<1`
	// amd64: `MOVW(L|Q)?ZX\s*\([A-Z0-9]+\)\([A-Z0-9]+\*2\)`
	return p[7*x]
}

//go:noinline
func store7(p []uint32, x int, v uint32) {
	// arm64: `MOVW\s*R[0-9]+,\s*\(R[0-9]+\)\(R[0-9]+<<2\)` -`SUB\s*R[0-9]+<<2`
	// amd64: `MOVL\s*[A-Z0-9]+,\s*\([A-Z0-9]+\)\([A-Z0-9]+\*4\)`
	p[7*x] = v
}

//go:noinline
func cse7(p []uint32, x int) uint32 {
	// arm64: `MOVWU\s*\(R[0-9]+\)\(R[0-9]+<<2\)` -`SUB\s*R[0-9]+<<2`
	// amd64: `MOVL\s*\([A-Z0-9]+\)\([A-Z0-9]+\*4\)`
	return p[7*x] + p[7*x+1]
}

//go:noinline
func genMath(x int) int {
	// amd64: `IMUL3?Q\s*[$]28` -`SHLQ\s*[$]2`
	// arm64: `SUB\s*R[0-9]+<<2` `LSL\s*[$]5` -`LSL\s*[$]2`
	// riscv64: `(MUL|MADD)` -`SLLI\s*.*,\s*2`
	return 4 * (7 * x)
}

