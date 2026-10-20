// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"math/bits"
	"unsafe"
)

// integer is the set of predeclared integer types and their defined
// counterparts. MulLen checks products over every one of them, so a
// make length of any integer type keeps its exact overflow behavior.
type integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

// MulLen returns a*b for a make length or capacity (SPEC T-303). It
// panics with `logos: make length overflow: <expr>` when the exact
// product does not fit the operand type, so an overflowing length never
// allocates a wrapped small buffer. The lowering passes the source text
// of the product as expr. Go packages keep the wrapped upstream result;
// only lowered .lgo make arguments call this helper.
func MulLen[T integer](a, b T, expr string) T {
	if mulOverflows(a, b) {
		panic("logos: make length overflow: " + expr)
	}
	return a * b
}

// mulOverflows reports whether the exact product of a and b lies outside
// the range of T. A zero wrapped minus one is positive only for unsigned
// types, so the test selects the unsigned or the signed check.
func mulOverflows[T integer](a, b T) bool {
	var zero T
	bitsOf := uint(unsafe.Sizeof(zero)) * 8
	if zero-1 > zero {
		hi, lo := bits.Mul64(uint64(a), uint64(b))
		return hi != 0 || (bitsOf < 64 && lo>>bitsOf != 0)
	}
	return signedMulOverflows(a, b, bitsOf)
}

// signedMulOverflows reports whether the exact product of two signed
// values of bitsOf bits overflows. It multiplies magnitudes, because the
// most negative value has no representable negation: a negative product
// fits while its magnitude is at most 2^(bitsOf-1), and a non-negative
// one while it is below 2^(bitsOf-1).
func signedMulOverflows[T integer](a, b T, bitsOf uint) bool {
	hi, lo := bits.Mul64(magnitude(a), magnitude(b))
	if hi != 0 {
		return true
	}
	limit := uint64(1) << (bitsOf - 1)
	if (a < 0) != (b < 0) {
		return lo > limit
	}
	return lo >= limit
}

// magnitude returns |v| as an unsigned 64-bit number. It negates v+1
// before adding one back, so the most negative value stays exact.
func magnitude[T integer](v T) uint64 {
	if v < 0 {
		return uint64(-(v + 1)) + 1
	}
	return uint64(v)
}
