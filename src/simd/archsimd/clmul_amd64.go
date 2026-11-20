// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package archsimd

import (
	"internal/cpu"
	"math/bits"
)

// The carryless multiply instruction on Uint64x2 (VPCLMULQDQ with 128-bit XMM
// registers) requires the PCLMULQDQ CPU feature, and on Uint64x4 and Uint64x8
// (256- and 512-bit registers) it requires the VPCLMULQDQ CPU feature. Many
// CPUs have vectors wider than their carryless multiply: for example, CPUs
// with AVX2 before Ice Lake and Zen 3 lack VPCLMULQDQ, as do the first few
// generations of CPUs with AVX-512.
//
// To make carryless multiply available whenever its vector type is, each
// CarrylessMul method uses the instruction for its own vector type if the
// CPU supports it, and otherwise applies the Uint64x2 instruction to each
// 128-bit part of its operands. On the rare CPU that has AVX but lacks
// PCLMULQDQ, it falls back to software.
//
// Each method tests a single CPU feature flag for each of these levels, since
// internal/cpu ensures that each wider form of the instruction implies the
// narrower ones.
//
// The xyHiLo immediate of the carrylessMultiply intrinsic must be a constant
// for it to compile to a single instruction, so each method spells out its
// immediate rather than sharing these steps through helper methods.

// CarrylessMulEven computes the elementwise carryless multiplication of
// even-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i])
//
// A carryless multiplication uses bitwise XOR instead of add-with-carry.
// For example, to compute the carryless multiply of 0b1110 and 0b1011:
//
//	     1110
//	   ⊗ 1011
//	  ───────
//	     1110
//	    1110
//	   0000
//	⊕ 1110
//	  ───────
//	  1100010
//
// Carryless multiply can also be viewed as multiplying polynomials with
// coefficients from GF(2). For example, the above example can be represented as
//
//	  (x^3 + x^2 + x^1) * (x^3 + x^1 + x^0)
//	= (x^6 + x^5 + (1^1)x^4 + (1^1)x^3 + (1^1)x^2 + x^1)
//	= (x^6 + x^5 + x^1)
//
// Asm: VPCLMULQDQ if CPU feature AVXCLMUL, Emulated otherwise, CPU Feature: AVX
func (x Uint64x2) CarrylessMulEven(y Uint64x2) (z Uint64x2) {
	if cpu.X86.HasPCLMULQDQ {
		return x.carrylessMultiply(0x00, y)
	}
	return x.carrylessMultiplySoft(0x00, y)
}

// CarrylessMulOdd computes the elementwise carryless multiplication of
// odd-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i+1])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVXCLMUL, Emulated otherwise, CPU Feature: AVX
func (x Uint64x2) CarrylessMulOdd(y Uint64x2) (z Uint64x2) {
	if cpu.X86.HasPCLMULQDQ {
		return x.carrylessMultiply(0x11, y)
	}
	return x.carrylessMultiplySoft(0x11, y)
}

// CarrylessMulOddEven computes the elementwise carryless multiplication of
// odd-indexed elements of x and even-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVXCLMUL, Emulated otherwise, CPU Feature: AVX
func (x Uint64x2) CarrylessMulOddEven(y Uint64x2) (z Uint64x2) {
	if cpu.X86.HasPCLMULQDQ {
		return x.carrylessMultiply(0x01, y)
	}
	return x.carrylessMultiplySoft(0x01, y)
}

// CarrylessMulEvenOdd computes the elementwise carryless multiplication of
// even-indexed elements of x and odd-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i+1])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVXCLMUL, Emulated otherwise, CPU Feature: AVX
func (x Uint64x2) CarrylessMulEvenOdd(y Uint64x2) (z Uint64x2) {
	if cpu.X86.HasPCLMULQDQ {
		return x.carrylessMultiply(0x10, y)
	}
	return x.carrylessMultiplySoft(0x10, y)
}

// CarrylessMulEven computes the elementwise carryless multiplication of
// even-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i])
//
// A carryless multiplication uses bitwise XOR instead of add-with-carry.
// For example, to compute the carryless multiply of 0b1110 and 0b1011:
//
//	     1110
//	   ⊗ 1011
//	  ───────
//	     1110
//	    1110
//	   0000
//	⊕ 1110
//	  ───────
//	  1100010
//
// Carryless multiply can also be viewed as multiplying polynomials with
// coefficients from GF(2). For example, the above example can be represented as
//
//	  (x^3 + x^2 + x^1) * (x^3 + x^1 + x^0)
//	= (x^6 + x^5 + (1^1)x^4 + (1^1)x^3 + (1^1)x^2 + x^1)
//	= (x^6 + x^5 + x^1)
//
// Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
func (x Uint64x4) CarrylessMulEven(y Uint64x4) (z Uint64x4) {
	if cpu.X86.HasVPCLMULQDQ {
		return x.carrylessMultiply(0x00, y)
	}
	if cpu.X86.HasPCLMULQDQ {
		lo := x.Lo().carrylessMultiply(0x00, y.Lo())
		hi := x.Hi().carrylessMultiply(0x00, y.Hi())
		return lo.Enlarge().SetHi(hi)
	}
	return x.carrylessMultiplySoft(0x00, y)
}

// CarrylessMulOdd computes the elementwise carryless multiplication of
// odd-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i+1])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
func (x Uint64x4) CarrylessMulOdd(y Uint64x4) (z Uint64x4) {
	if cpu.X86.HasVPCLMULQDQ {
		return x.carrylessMultiply(0x11, y)
	}
	if cpu.X86.HasPCLMULQDQ {
		lo := x.Lo().carrylessMultiply(0x11, y.Lo())
		hi := x.Hi().carrylessMultiply(0x11, y.Hi())
		return lo.Enlarge().SetHi(hi)
	}
	return x.carrylessMultiplySoft(0x11, y)
}

// CarrylessMulOddEven computes the elementwise carryless multiplication of
// odd-indexed elements of x and even-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
func (x Uint64x4) CarrylessMulOddEven(y Uint64x4) (z Uint64x4) {
	if cpu.X86.HasVPCLMULQDQ {
		return x.carrylessMultiply(0x01, y)
	}
	if cpu.X86.HasPCLMULQDQ {
		lo := x.Lo().carrylessMultiply(0x01, y.Lo())
		hi := x.Hi().carrylessMultiply(0x01, y.Hi())
		return lo.Enlarge().SetHi(hi)
	}
	return x.carrylessMultiplySoft(0x01, y)
}

// CarrylessMulEvenOdd computes the elementwise carryless multiplication of
// even-indexed elements of x and odd-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i+1])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVX2CLMUL, Emulated otherwise, CPU Feature: AVX2
func (x Uint64x4) CarrylessMulEvenOdd(y Uint64x4) (z Uint64x4) {
	if cpu.X86.HasVPCLMULQDQ {
		return x.carrylessMultiply(0x10, y)
	}
	if cpu.X86.HasPCLMULQDQ {
		lo := x.Lo().carrylessMultiply(0x10, y.Lo())
		hi := x.Hi().carrylessMultiply(0x10, y.Hi())
		return lo.Enlarge().SetHi(hi)
	}
	return x.carrylessMultiplySoft(0x10, y)
}

// CarrylessMulEven computes the elementwise carryless multiplication of
// even-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i])
//
// A carryless multiplication uses bitwise XOR instead of add-with-carry.
// For example, to compute the carryless multiply of 0b1110 and 0b1011:
//
//	     1110
//	   ⊗ 1011
//	  ───────
//	     1110
//	    1110
//	   0000
//	⊕ 1110
//	  ───────
//	  1100010
//
// Carryless multiply can also be viewed as multiplying polynomials with
// coefficients from GF(2). For example, the above example can be represented as
//
//	  (x^3 + x^2 + x^1) * (x^3 + x^1 + x^0)
//	= (x^6 + x^5 + (1^1)x^4 + (1^1)x^3 + (1^1)x^2 + x^1)
//	= (x^6 + x^5 + x^1)
//
// Asm: VPCLMULQDQ if CPU feature AVX512CLMUL, Emulated otherwise, CPU Feature: AVX512
func (x Uint64x8) CarrylessMulEven(y Uint64x8) (z Uint64x8) {
	if cpu.X86.HasAVX512VPCLMULQDQ {
		return x.carrylessMultiply(0x00, y)
	}
	// There is no Uint64x4 (256-bit) CLMUL feature separate from Uint64x8
	// CLMUL, so try falling back to Uint64x2 (128-bit) CLMUL.
	if cpu.X86.HasPCLMULQDQ {
		xlo, xhi, ylo, yhi := x.Lo(), x.Hi(), y.Lo(), y.Hi()
		z0 := xlo.Lo().carrylessMultiply(0x00, ylo.Lo())
		z1 := xlo.Hi().carrylessMultiply(0x00, ylo.Hi())
		z2 := xhi.Lo().carrylessMultiply(0x00, yhi.Lo())
		z3 := xhi.Hi().carrylessMultiply(0x00, yhi.Hi())
		return z0.Enlarge().SetHi(z1).Enlarge().SetHi(z2.Enlarge().SetHi(z3))
	}
	return x.carrylessMultiplySoft(0x00, y)
}

// CarrylessMulOdd computes the elementwise carryless multiplication of
// odd-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i+1])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVX512CLMUL, Emulated otherwise, CPU Feature: AVX512
func (x Uint64x8) CarrylessMulOdd(y Uint64x8) (z Uint64x8) {
	if cpu.X86.HasAVX512VPCLMULQDQ {
		return x.carrylessMultiply(0x11, y)
	}
	if cpu.X86.HasPCLMULQDQ {
		xlo, xhi, ylo, yhi := x.Lo(), x.Hi(), y.Lo(), y.Hi()
		z0 := xlo.Lo().carrylessMultiply(0x11, ylo.Lo())
		z1 := xlo.Hi().carrylessMultiply(0x11, ylo.Hi())
		z2 := xhi.Lo().carrylessMultiply(0x11, yhi.Lo())
		z3 := xhi.Hi().carrylessMultiply(0x11, yhi.Hi())
		return z0.Enlarge().SetHi(z1).Enlarge().SetHi(z2.Enlarge().SetHi(z3))
	}
	return x.carrylessMultiplySoft(0x11, y)
}

// CarrylessMulOddEven computes the elementwise carryless multiplication of
// odd-indexed elements of x and even-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVX512CLMUL, Emulated otherwise, CPU Feature: AVX512
func (x Uint64x8) CarrylessMulOddEven(y Uint64x8) (z Uint64x8) {
	if cpu.X86.HasAVX512VPCLMULQDQ {
		return x.carrylessMultiply(0x01, y)
	}
	if cpu.X86.HasPCLMULQDQ {
		xlo, xhi, ylo, yhi := x.Lo(), x.Hi(), y.Lo(), y.Hi()
		z0 := xlo.Lo().carrylessMultiply(0x01, ylo.Lo())
		z1 := xlo.Hi().carrylessMultiply(0x01, ylo.Hi())
		z2 := xhi.Lo().carrylessMultiply(0x01, yhi.Lo())
		z3 := xhi.Hi().carrylessMultiply(0x01, yhi.Hi())
		return z0.Enlarge().SetHi(z1).Enlarge().SetHi(z2.Enlarge().SetHi(z3))
	}
	return x.carrylessMultiplySoft(0x01, y)
}

// CarrylessMulEvenOdd computes the elementwise carryless multiplication of
// even-indexed elements of x and odd-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i+1])
//
// See [CarrylessMulEven] for details about carryless multiply.
//
// Asm: VPCLMULQDQ if CPU feature AVX512CLMUL, Emulated otherwise, CPU Feature: AVX512
func (x Uint64x8) CarrylessMulEvenOdd(y Uint64x8) (z Uint64x8) {
	if cpu.X86.HasAVX512VPCLMULQDQ {
		return x.carrylessMultiply(0x10, y)
	}
	if cpu.X86.HasPCLMULQDQ {
		xlo, xhi, ylo, yhi := x.Lo(), x.Hi(), y.Lo(), y.Hi()
		z0 := xlo.Lo().carrylessMultiply(0x10, ylo.Lo())
		z1 := xlo.Hi().carrylessMultiply(0x10, ylo.Hi())
		z2 := xhi.Lo().carrylessMultiply(0x10, yhi.Lo())
		z3 := xhi.Hi().carrylessMultiply(0x10, yhi.Hi())
		return z0.Enlarge().SetHi(z1).Enlarge().SetHi(z2.Enlarge().SetHi(z3))
	}
	return x.carrylessMultiplySoft(0x10, y)
}

// The carrylessMultiplySoft methods are all marked "//go:noinline" to ensure
// that the hardware fast paths of the CarrylessMul* methods are small
// enough to inline.

// carrylessMultiplySoft is a software implementation of x.carrylessMultiply(xyHiLo, y).
//
//go:noinline
func (x Uint64x2) carrylessMultiplySoft(xyHiLo uint8, y Uint64x2) Uint64x2 {
	var xs, ys, zs [2]uint64
	x.StoreArray(&xs)
	y.StoreArray(&ys)
	carrylessMultiplySoft(zs[:], xs[:], ys[:], xyHiLo)
	return LoadUint64x2Array(&zs)
}

// carrylessMultiplySoft is a software implementation of x.carrylessMultiply(xyHiLo, y).
//
//go:noinline
func (x Uint64x4) carrylessMultiplySoft(xyHiLo uint8, y Uint64x4) Uint64x4 {
	var xs, ys, zs [4]uint64
	x.StoreArray(&xs)
	y.StoreArray(&ys)
	carrylessMultiplySoft(zs[:], xs[:], ys[:], xyHiLo)
	return LoadUint64x4Array(&zs)
}

// carrylessMultiplySoft is a software implementation of x.carrylessMultiply(xyHiLo, y).
//
//go:noinline
func (x Uint64x8) carrylessMultiplySoft(xyHiLo uint8, y Uint64x8) Uint64x8 {
	var xs, ys, zs [8]uint64
	x.StoreArray(&xs)
	y.StoreArray(&ys)
	carrylessMultiplySoft(zs[:], xs[:], ys[:], xyHiLo)
	return LoadUint64x8Array(&zs)
}

// carrylessMultiplySoft sets each pair of elements of z to the 128-bit
// carryless product of one element from the corresponding pair of x and one
// from the corresponding pair of y. As for the carrylessMultiply intrinsic,
// bit 0 of xyHiLo selects the low (0) or high (1) element of each pair of x,
// and bit 4 does the same for y.
func carrylessMultiplySoft(z, x, y []uint64, xyHiLo uint8) {
	xi, yi := int(xyHiLo&1), int(xyHiLo>>4&1)
	for i := 0; i < len(z); i += 2 {
		z[i], z[i+1] = carrylessMultiply64(x[i+xi], y[i+yi])
	}
}

// carrylessMultiply64 returns the 128-bit carryless product of x and y. It is
// constant-time, assuming that integer multiplication is.
//
// It uses integer multiplication with "holes": it splits each operand into
// five parts, where part i holds the bits whose position is i mod 5, so that
// the set bits of each part are separated by at least four zeros. In the
// integer product of two parts, the bits whose position is (i+j) mod 5 are
// then sums of at most 13 one-bit products, which fit in the four bits of the
// hole and so never carry into each other. The low bit of each sum is the
// carryless product bit. See
// https://timtaubert.de/blog/2017/06/verified-binary-multiplication-for-ghash/.
func carrylessMultiply64(x, y uint64) (lo, hi uint64) {
	const (
		// mi selects the bits whose position is i mod 5. Since 64 is 4
		// mod 5, in the high half of a 128-bit value, m(i+1) selects
		// those bits instead.
		m0 = 0x1084210842108421
		m1 = 0x2108421084210842
		m2 = 0x4210842108421084
		m3 = 0x8421084210842108
		m4 = 0x0842108421084210
	)
	x0, x1, x2, x3, x4 := x&m0, x&m1, x&m2, x&m3, x&m4
	y0, y1, y2, y3, y4 := y&m0, y&m1, y&m2, y&m3, y&m4

	// mul5 returns the XOR of the five integer products ai*bi. In the
	// products for result part k, the indexes of each ai and bi sum to
	// k mod 5.
	mul5 := func(a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 uint64) (lo, hi uint64) {
		h0, l0 := bits.Mul64(a0, b0)
		h1, l1 := bits.Mul64(a1, b1)
		h2, l2 := bits.Mul64(a2, b2)
		h3, l3 := bits.Mul64(a3, b3)
		h4, l4 := bits.Mul64(a4, b4)
		return l0 ^ l1 ^ l2 ^ l3 ^ l4, h0 ^ h1 ^ h2 ^ h3 ^ h4
	}
	l0, h0 := mul5(x0, y0, x1, y4, x2, y3, x3, y2, x4, y1)
	l1, h1 := mul5(x0, y1, x1, y0, x2, y4, x3, y3, x4, y2)
	l2, h2 := mul5(x0, y2, x1, y1, x2, y0, x3, y4, x4, y3)
	l3, h3 := mul5(x0, y3, x1, y2, x2, y1, x3, y0, x4, y4)
	l4, h4 := mul5(x0, y4, x1, y3, x2, y2, x3, y1, x4, y0)

	lo = l0&m0 | l1&m1 | l2&m2 | l3&m3 | l4&m4
	hi = h0&m1 | h1&m2 | h2&m3 | h3&m4 | h4&m0
	return lo, hi
}
