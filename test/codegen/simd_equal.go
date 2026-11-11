// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of == and != on 128-bit SIMD vectors and
// masks, which every architecture has.

//go:build goexperiment.simd

package codegen

import "simd/archsimd"

// Integer vectors and masks are equal if their XOR is zero.

func eqInt32x4(x, y archsimd.Int32x4) bool {
	// amd64:`VPXOR` `VPTEST` `SETEQ` -`CALL`
	// arm64:`VEOR` `VUMAXV` -`CALL`
	// wasm:`V128Xor` `V128AnyTrue` `I64Eqz` -`CALL`
	return x == y
}

func neUint8x16(x, y archsimd.Uint8x16) bool {
	// amd64:`VPXOR` `VPTEST` `SETNE` -`CALL`
	// arm64:`VEOR` `VUMAXV` -`CALL`
	// wasm:`V128Xor` `V128AnyTrue` -`CALL`
	return x != y
}

func eqMask64x2(x, y archsimd.Mask64x2) bool {
	// amd64:`VPXOR` `VPTEST` `SETEQ` -`CALL`
	// arm64:`VEOR` `VUMAXV` -`CALL`
	// wasm:`V128Xor` `V128AnyTrue` `I64Eqz` -`CALL`
	return x == y
}

// Float vectors are equal if no element compares not-equal.

func eqFloat32x4(x, y archsimd.Float32x4) bool {
	// amd64:`VCMPPS` `VPTEST` `SETEQ` -`CALL`
	// arm64:`VFCMEQ` `VNOT` `VUMAXV` -`CALL`
	// wasm:`F32x4Ne` `V128AnyTrue` `I64Eqz` -`CALL`
	return x == y
}

func neFloat64x2(x, y archsimd.Float64x2) bool {
	// amd64:`VCMPPD` `VPTEST` `SETNE` -`CALL`
	// arm64:`VFCMEQ` `VNOT` `VUMAXV` -`CALL`
	// wasm:`F64x2Ne` `V128AnyTrue` -`CALL`
	return x != y
}
