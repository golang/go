// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of amd64 SIMD carryless multiply.

//go:build goexperiment.simdstable && amd64

package codegen

import "simd/archsimd"

// The CarrylessMul methods check CPU features to choose between a single
// instruction for the whole vector, the Uint64x2 instruction applied to each
// 128-bit part, and software. Check that the methods are inlined, that both
// instruction paths are inlined with a constant immediate, and that only the
// software path is a call.

func simdCarrylessMulUint64x2(x, y archsimd.Uint64x2) archsimd.Uint64x2 {
	// amd64:`VPCLMULQDQ \$0, X[0-9]+, X[0-9]+, X[0-9]+` -`CALL .*\.CarrylessMul` `CALL .*\.carrylessMultiplySoft`
	return x.CarrylessMulEven(y)
}

func simdCarrylessMulUint64x4(x, y archsimd.Uint64x4) archsimd.Uint64x4 {
	// amd64:`VPCLMULQDQ \$17, Y[0-9]+, Y[0-9]+, Y[0-9]+` `VPCLMULQDQ \$17, X[0-9]+, X[0-9]+, X[0-9]+`
	// amd64:`VEXTRACTI128` `VINSERTI128` -`CALL .*\.CarrylessMul` `CALL .*\.carrylessMultiplySoft`
	return x.CarrylessMulOdd(y)
}

func simdCarrylessMulUint64x8(x, y archsimd.Uint64x8) archsimd.Uint64x8 {
	// amd64:`VPCLMULQDQ \$16, Z[0-9]+, Z[0-9]+, Z[0-9]+` `VPCLMULQDQ \$16, X[0-9]+, X[0-9]+, X[0-9]+`
	// amd64:`VEXTRACTI64X4` `VINSERTI64X4` -`CALL .*\.CarrylessMul` `CALL .*\.carrylessMultiplySoft`
	return x.CarrylessMulEvenOdd(y)
}

// When the Uint64x2 instruction stands in for a wider one on a CPU with AVX-512,
// it must use registers X0 through X15: with X16 through X31 it would need the
// EVEX encoding, which requires the wider VPCLMULQDQ feature that's missing.
//
// The register allocator prefers the low registers, so this only tests
// something if it runs out of them. Keep 16 other vectors live across the
// multiply, and check that some of them are in the high registers.

func simdCarrylessMulUint64x8VEX(p *[18]archsimd.Uint64x8) archsimd.Uint64x8 {
	v0, v1, v2, v3 := p[0], p[1], p[2], p[3]
	v4, v5, v6, v7 := p[4], p[5], p[6], p[7]
	v8, v9, v10, v11 := p[8], p[9], p[10], p[11]
	// amd64:`VMOVDQU64 .*, Z(1[6-9]|2[0-9]|3[01])`
	v12, v13, v14, v15 := p[12], p[13], p[14], p[15]
	// amd64:-`VPCLMULQDQ \$1, .*X(1[6-9]|2[0-9]|3[01])`
	r := p[16].CarrylessMulOddEven(p[17])
	return r.Xor(v0).Xor(v1).Xor(v2).Xor(v3).
		Xor(v4).Xor(v5).Xor(v6).Xor(v7).
		Xor(v8).Xor(v9).Xor(v10).Xor(v11).
		Xor(v12).Xor(v13).Xor(v14).Xor(v15)
}
