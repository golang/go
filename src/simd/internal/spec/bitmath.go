// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//simdgen:category Bit math

package spec

import "math/bits"

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
func OnesCount[E Ints | Uints, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		switch any(x).(type) {
		case int8, uint8:
			return E(bits.OnesCount8(uint8(x)))
		case int16, uint16:
			return E(bits.OnesCount16(uint16(x)))
		case int32, uint32:
			return E(bits.OnesCount32(uint32(x)))
		case int64, uint64:
			return E(bits.OnesCount64(uint64(x)))
		}
		panic("unreachable")
	})
}

// TODO: Should LeadingZeros be defined on signed types?

// LeadingZeros counts the leading zero bits of each element in x,
// starting from the most significant bit.
// If an element is 0, the result is {{.xN}}.
//
//	z[i] = bits.LeadingZeros(x[i])
func LeadingZeros[E Ints | Uints, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		switch any(x).(type) {
		case int8, uint8:
			return E(bits.LeadingZeros8(uint8(x)))
		case int16, uint16:
			return E(bits.LeadingZeros16(uint16(x)))
		case int32, uint32:
			return E(bits.LeadingZeros32(uint32(x)))
		case int64, uint64:
			return E(bits.LeadingZeros64(uint64(x)))
		}
		panic("unreachable")
	})
}

// TODO: Right now, nothing supports TrailingZeros.

// TrailingZeros counts the trailing zero bits of each element in x,
// starting from the least significant bit.
// If an element is 0, the result is {{.xN}}.
//
//	z[i] = bits.TrailingZeros(x[i])
func TrailingZeros[E Uints, W Width](x Vec[E, W]) (z Vec[E, W]) {
	return map1[E, W, E, W](x, func(x E) E {
		switch x := any(x).(type) {
		case uint8:
			return E(bits.TrailingZeros8(x))
		case uint16:
			return E(bits.TrailingZeros16(x))
		case uint32:
			return E(bits.TrailingZeros32(x))
		case uint64:
			return E(bits.TrailingZeros64(x))
		}
		panic("unreachable")
	})
}

// TODO: Calling this "Trailing zeros" for masks breaks the abstraction boundary
// around the exact mask representation. E.g., what does it mean for a wide mask
// or a byte mask?

// TrailingZeros returns the number of low-order false (zero) elements in mask
// x. If the mask is entirely false, it returns {{lanes .x "x"}}.
//
//specgen:name TrailingZeros
func TrailingZerosMask[E MaskElt, W Width](x Vec[E, W]) int {
	i := 0
	for ; i < len(x); i++ {
		if x[i] != 0 {
			break
		}
	}
	return i
}

// CarrylessMultiplyEven computes the elementwise carryless multiplication of
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
//specgen:commutative
func CarrylessMultiplyEven[E Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return carrylessMultiplyPair(x, y, false, false)
}

// CarrylessMultiplyEvenOdd computes the elementwise carryless multiplication of
// even-indexed elements of x and odd-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i+1])
//
// See [CarrylessMultiplyEven] for details about carryless multiply.
//
//specgen:commutative
func CarrylessMultiplyEvenOdd[E Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return carrylessMultiplyPair(x, y, false, true)
}

// CarrylessMultiplyOdd computes the elementwise carryless multiplication of
// odd-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i+1])
//
// See [CarrylessMultiplyEven] for details about carryless multiply.
//
//specgen:commutative
func CarrylessMultiplyOdd[E Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return carrylessMultiplyPair(x, y, true, true)
}

// CarrylessMultiplyOddEven computes the elementwise carryless multiplication of
// odd-indexed elements of x and even-indexed elements of y. The result of each
// carryless multiply is twice the width of the input elements. The high bits
// are stored in elements z[2*i+1] and the low bits are stored in elements
// z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i])
//
// See [CarrylessMultiplyEven] for details about carryless multiply.
//
//specgen:commutative
func CarrylessMultiplyOddEven[E Uints, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
	return carrylessMultiplyPair(x, y, true, false)
}

func carrylessMultiplyPair[E Uints, W Width](x, y Vec[E, W], xOdd, yOdd bool) (z Vec[E, W]) {
	z = makeVec[E, W]()
	pairs := z.len() / 2
	for i := 0; i < pairs; i++ {
		xi := x[2*i]
		if xOdd {
			xi = x[2*i+1]
		}
		yi := y[2*i]
		if yOdd {
			yi = y[2*i+1]
		}
		z[2*i+1], z[2*i] = clmul(xi, yi)
	}
	return z
}
