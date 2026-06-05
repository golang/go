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

func vpsumd(xlo, xhi, ylo, yhi uint64) (lo, hi uint64) {
	lo, hi = clmul64(xhi, yhi)
	l, h := clmul64(xlo, ylo)
	hi ^= h
	lo ^= l
	return
}

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

func em2(xlo, xhi, ylo, yhi uint64) string {
	lx := newT(xlo, 0)
	ly := newT(ylo, 0)
	hx := newT(xhi, 0)
	hy := newT(yhi, 0)

	z := (lx.ClMul(ly)).Xor(hx.ClMul(hy))

	return fmt.Sprintf("0x%08x%08x", z.b, z.a)
}

func em1(a, b, c, d uint64) string {
	lo, hi := vpsumd(a, b, c, d)
	return fmt.Sprintf("0x%08x%08x", hi, lo)
}

func set0(v uint64) simd.Uint64s {
	a := [2]uint64{v, 0}
	r, _ := simd.LoadUint64sPart(a[:])
	return r
}

func get(v simd.Uint64s) (lo, hi uint64) {
	var a [2]uint64
	v.StorePart(a[:])
	return a[0], a[1]
}

func em3(xlo, xhi, ylo, yhi uint64) string {
	lx := set0(xlo)
	ly := set0(ylo)
	hx := set0(xhi)
	hy := set0(yhi)

	z := (lx.CarrylessMultiplyEven(ly)).Xor(hx.CarrylessMultiplyEven(hy))

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

	fmt.Println("EMULATION 1")
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, x, 1, 16, em1(x, x, 1, 16))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, y, 1, 16, em1(x, y, 1, 16))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, y, x, y, em1(x, y, x, y))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", 1, 2, y*4, y, em1(1, 2, y*4, y))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", a, b, c, d, em1(a, b, c, d))

	fmt.Println("EMULATION 2")
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, x, 1, 16, em2(x, x, 1, 16))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, y, 1, 16, em2(x, y, 1, 16))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, y, x, y, em2(x, y, x, y))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", 1, 2, y*4, y, em2(1, 2, y*4, y))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", a, b, c, d, em2(a, b, c, d))

	fmt.Println("EMULATION 3")
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, x, 1, 16, em3(x, x, 1, 16))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, y, 1, 16, em3(x, y, 1, 16))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", x, y, x, y, em3(x, y, x, y))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", 1, 2, y*4, y, em3(1, 2, y*4, y))
	fmt.Printf("clmul(0x%08x, 0x%08x, 0x%08x, 0x%08x) = %s\n", a, b, c, d, em3(a, b, c, d))

	for i := range 10000 {
		a, b, c, d := rand.Uint64(), rand.Uint64(), rand.Uint64(), rand.Uint64()

		e1 := em1(a, b, c, d)
		e2 := em2(a, b, c, d)
		e3 := em3(a, b, c, d)

		if e1 != e2 || e1 != e3 {
			t.Errorf("Mismatch at %d, a,b,c,d = 0x%08x, 0x%08x, 0x%08x, 0x%08x; e1=%s, e2=%s, e3=%s", i, a, b, c, d, e1, e2, e3)
			if i > 5 {
				return
			}
		}

	}

}

type T struct {
	a, b uint64
}

func newT(lo, hi uint64) T {
	return T{a: lo, b: hi}
}

func (x T) And(y T) T {
	return T{a: x.a & y.a, b: x.b & y.b}
}

func (x T) Xor(y T) T {
	return T{a: x.a ^ y.a, b: x.b ^ y.b}
}

func (x T) Or(y T) T {
	return T{a: x.a | y.a, b: x.b | y.b}
}

func (x T) MWL(y T) T { // MulWidenLo
	hi, lo := bits.Mul64(x.a, y.a)
	return T{a: lo, b: hi}
}

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
