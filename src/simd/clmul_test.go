// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"fmt"
	"math/bits"
	"math/rand"
	"simd"
	"testing"
)

// vpsumd returns the 128-bit result of
// clmul(xlo,ylo)^clmul(xhi,yhi)
// using a plain and obvious implementation
// of clmul.
// "vpsumd" is the name of the Power PC 64-bit
// instruction with these same semantics.
func vpsumd(xlo, xhi, ylo, yhi uint64) (lo, hi uint64) {
	lo, hi = clmul64(xhi, yhi)
	l, h := clmul64(xlo, ylo)
	hi ^= h
	lo ^= l
	return
}

// clmul64 is a plain and obvious implementation of
// carryless multiply.
func clmul64(a, b uint64) (lo, hi uint64) {
	for i := range uint(64) {
		if (a>>i)&1 == 1 {
			if i == 0 {
				lo ^= b
			} else {
				lo ^= b << i
				hi ^= b >> (64 - i)
			}
		}
	}
	return
}

// em1 returns the string representation of
//
//	clmul(a,c)^clmul(b,d)
//
// using a plain and obvious implementation of carryless multiply.
func em1(a, b, c, d uint64) string {
	lo, hi := vpsumd(a, b, c, d)
	return fmt.Sprintf("0x%08x%08x", hi, lo)
}

// em1 returns the string representation of
//
//	clmul(xlo,ylo)^clmul(xhi,yhi)
//
// using a clever constant-time implementation of clmul
// using simpler simd instructions, for an emulated simd
// type.
func em2(xlo, xhi, ylo, yhi uint64) string {
	lx := newT(xlo, 0)
	ly := newT(ylo, 0)
	hx := newT(xhi, 0)
	hy := newT(yhi, 0)

	z := (lx.ClMul(ly)).Xor(hx.ClMul(hy))

	return fmt.Sprintf("0x%08x%08x", z.b, z.a)
}

// set0 returns a vector of uint64s that is zero
// except for element 0 which is initialized
// to v.
func set0(v uint64) simd.Uint64s {
	a := [2]uint64{v, 0}
	r, _ := simd.LoadUint64sPart(a[:])
	return r
}

// set1 returns a vector of uint64s that is zero
// except for element 1 which is initialized
// to v.
func set1(v uint64) simd.Uint64s {
	a := [2]uint64{0, v}
	r, _ := simd.LoadUint64sPart(a[:])
	return r
}

// get returns the 0 and 1 elements of a vector
// of uint64s.
func get(v simd.Uint64s) (lo, hi uint64) {
	var a [2]uint64
	v.StorePart(a[:])
	return a[0], a[1]
}

// em3 returns the string representation of
//
//	clmul(xlo,ylo)^clmul(xhi,yhi)
//
// using the supplied simd operation
// CarrylessMultiplyEven
func em3(xlo, xhi, ylo, yhi uint64) string {
	lx := set0(xlo)
	ly := set0(ylo)
	hx := set0(xhi)
	hy := set0(yhi)

	z := (lx.CarrylessMultiplyEven(ly)).Xor(hx.CarrylessMultiplyEven(hy))

	lo, hi := get(z)
	return fmt.Sprintf("0x%08x%08x", hi, lo)
}

// em3 returns the string representation of
//
//	clmul(xlo,ylo)^clmul(xhi,yhi)
//
// using the supplied simd operation
// CarrylessMultiplyOdd
func em4(xlo, xhi, ylo, yhi uint64) string {
	lx := set1(xlo)
	ly := set1(ylo)
	hx := set1(xhi)
	hy := set1(yhi)

	z := (lx.CarrylessMultiplyOdd(ly)).Xor(hx.CarrylessMultiplyOdd(hy))

	lo, hi := get(z)
	return fmt.Sprintf("0x%08x%08x", hi, lo)
}

func TestClMul(t *testing.T) {
	fmt.Println("Vector length:", simd.VectorBitSize())
	fmt.Println("Emulated:", simd.Emulated())
	fmt.Println("HasHWCLMUL:", simd.HasHardwareCarrylessMultiply())

	x := uint64(0x0807060504030201)
	y := uint64(0x0101010101010101)

	var a, b, c, d uint64
	a, b, c, d = 0x66b32838754f59a3, 0xaeba319ab2418c50, 0x45678b3c7f11fc73, 0xd62ef8ae5f7b693

	f := func(what string, f func(a, b, c, d uint64) string) {
		fmt.Println(what)
		fmt.Printf("vpsumd(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, x, 1, 16, f(x, x, 1, 16))
		fmt.Printf("vpsumd(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, y, 1, 16, f(x, y, 1, 16))
		fmt.Printf("vpsumd(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, y, x, y, f(x, y, x, y))
		fmt.Printf("vpsumd(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", 1, 2, y*4, y, f(1, 2, y*4, y))
		fmt.Printf("vpsumd(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", a, b, c, d, f(a, b, c, d))
	}

	f("Simple scalar clmul emulation", em1)
	f("Clever emulated vector clmul emulation", em2)
	f("CarrylessMultiplyEven", em3)
	f("CarrylessMultiplyOdd", em4)

	for i := range 10000 {
		a, b, c, d := rand.Uint64(), rand.Uint64(), rand.Uint64(), rand.Uint64()

		e1 := em1(a, b, c, d)
		e2 := em2(a, b, c, d)
		e3 := em3(a, b, c, d)
		e4 := em4(a, b, c, d)

		if e1 != e2 || e1 != e3 || e1 != e4 {
			t.Errorf("Mismatch at %d, a,b,c,d = 0x%08x, 0x%08x, 0x%08x, 0x%08x; e1=%s, e2=%s, e3=%s, e4=%s", i, a, b, c, d, e1, e2, e3, e4)
			if i > 5 {
				return
			}
		}
	}

}

// T is a simulated vector type
type T struct {
	a, b uint64
}

// newT returns a new vector (T)
// initialized with lower and upper
// 64-bit halves equal to lo and hi.
func newT(lo, hi uint64) T {
	return T{a: lo, b: hi}
}

// And returns the bitwise and of x and y
func (x T) And(y T) T {
	return T{a: x.a & y.a, b: x.b & y.b}
}

// Xor returns the bitwise xor of x and y
func (x T) Xor(y T) T {
	return T{a: x.a ^ y.a, b: x.b ^ y.b}
}

// Or returns the bitwise or of x and y
func (x T) Or(y T) T {
	return T{a: x.a | y.a, b: x.b | y.b}
}

// MWL returns the 128-bit unsigned product
// of x[0] times y[0].
func (x T) MWL(y T) T { // MulWidenLo
	hi, lo := bits.Mul64(x.a, y.a)
	return T{a: lo, b: hi}
}

// ClMul is a constant time implementation of carryless
// multiply low-parts implemented in terms of bitwise
// And, Or, and Xor, and MWL.
// MWL is a short name for MulWidenLow.
func (x T) ClMul(y T) T {
	m1 := newT(0x1084210842108421, 0x2108421084210842)
	m2 := newT(0x2108421084210842, 0x4210842108421084)
	m3 := newT(0x4210842108421084, 0x8421084210842108)
	m4 := newT(0x8421084210842108, 0x0842108421084210)
	m5 := newT(0x0842108421084210, 0x1084210842108421)

	x1 := x.And(m1)
	x2 := x.And(m2)
	x3 := x.And(m3)
	x4 := x.And(m4)
	x5 := x.And(m5)

	y1 := y.And(m1)
	y2 := y.And(m2)
	y3 := y.And(m3)
	y4 := y.And(m4)
	y5 := y.And(m5)

	// sum of x, y indices == K mod 5; mask index = K-1
	z := (x1.MWL(y1)).Xor(x2.MWL(y5)).Xor(x5.MWL(y2)).Xor(x3.MWL(y4)).Xor(x4.MWL(y3)).And(m1)
	z = (x4.MWL(y4)).Xor(x3.MWL(y5)).Xor(x5.MWL(y3)).Xor(x1.MWL(y2)).Xor(x2.MWL(y1)).And(m2).Or(z)
	z = (x2.MWL(y2)).Xor(x4.MWL(y5)).Xor(x5.MWL(y4)).Xor(x1.MWL(y3)).Xor(x3.MWL(y1)).And(m3).Or(z)
	z = (x5.MWL(y5)).Xor(x1.MWL(y4)).Xor(x4.MWL(y1)).Xor(x2.MWL(y3)).Xor(x3.MWL(y2)).And(m4).Or(z)
	z = (x3.MWL(y3)).Xor(x1.MWL(y5)).Xor(x5.MWL(y1)).Xor(x2.MWL(y4)).Xor(x4.MWL(y2)).And(m5).Or(z)

	return z
}
