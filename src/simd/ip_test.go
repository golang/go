// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"fmt"
	"math/rand/v2"
	"simd"
	"testing"
)

func fill(x, y []float32) {
	for i := range x {
		x[i] = 2*rand.Float32() - 1
		y[i] = 2*rand.Float32() - 1
	}
}

func checkErrors(b *testing.B, errors int) {
	b.Helper()
	if errors > 0 {
		b.Logf("errors = %d", errors)
	}
}

// BenchmarkIPFMA is simd vector inner product computing using FMA.
func BenchmarkIPFMA(b *testing.B) {
	x := make([]float32, ipBenchLen)
	y := make([]float32, ipBenchLen)

	fill(x, y)

	ip0, _, _ := ipFMA(x, y)

	var errors int
	for b.Loop() {
		z, _, _ := ipFMA(x, y)
		if z != ip0 {
			errors++
		}
	}
	checkErrors(b, errors)
}

func ipFMA(x, y []float32) (float32, int, bool) {
	var a simd.Float32s
	sumWidth := a.Len() * 32
	emulated := simd.Emulated()
	var i int
	for i = 0; i < len(x)-a.Len()+1; i += a.Len() {
		u := simd.LoadFloat32s(x[i : i+a.Len()])
		v := simd.LoadFloat32s(y[i : i+a.Len()])
		a = u.MulAdd(v, a)
	}
	if i < len(x) {
		a = first(simd.LoadFloat32sPart(x[i:])).MulAdd(
			first(simd.LoadFloat32sPart(y[i:])), a)
	}

	return sum(a), sumWidth, emulated
}

func TestIP(t *testing.T) {

	var a, b [50]float32
	for i := 0; i < 50; i++ {
		a[i] = float32(i)
		b[i] = float32(i)
	}
	x, sumWidth, emulated := ip(a[:50], b[:50])

	if x != 40425 {
		t.Errorf("Expected 40425, got %f", x)
	}

	fmt.Printf("ip: sum was computed in width %d, emulated = %v\n", sumWidth, emulated)
}

func TestIPGoTo(t *testing.T) {

	var a, b [50]float32
	for i := 0; i < 50; i++ {
		a[i] = float32(i)
		b[i] = float32(i)
	}
	x, sumWidth, emulated := ipGoTo(a[:50], b[:50])

	if x != 40425 {
		t.Errorf("Expected 40425, got %f", x)
	}

	fmt.Printf("ipgoto: sum was computed in width %d, emulated = %v\n", sumWidth, emulated)
}

func first[T, U any](t T, u U) T {
	return t
}

const ipBenchLen = 300000

// BenchmarkIP is simd vector inner product, vanilla transcription.
func BenchmarkIP(b *testing.B) {
	x := make([]float32, ipBenchLen)
	y := make([]float32, ipBenchLen)

	fill(x, y)

	ip0, _, _ := ip(x, y)

	var errors int
	for b.Loop() {
		z, _, _ := ip(x, y)
		if z != ip0 {
			errors++
		}
	}
	checkErrors(b, errors)
}

// BenchmarkIPUnroll is simd vector inner product, unrolled 4x vector ops.
func BenchmarkIPUnroll(b *testing.B) {
	x := make([]float32, ipBenchLen)
	y := make([]float32, ipBenchLen)

	fill(x, y)

	ip0, _, _ := ipU(x, y)

	var errors int
	for b.Loop() {
		z, _, _ := ipU(x, y)
		if z != ip0 {
			errors++
		}
	}
	checkErrors(b, errors)
}

// BenchmarkIPUnrollMore is simd vector inner product, unrolled 5x vector ops
func BenchmarkIPUnrollMore(b *testing.B) {
	x := make([]float32, ipBenchLen)
	y := make([]float32, ipBenchLen)

	fill(x, y)

	ip0, _, _ := ipUmore(x, y)

	var errors int
	for b.Loop() {
		z, _, _ := ipUmore(x, y)
		if z != ip0 {
			errors++
		}
	}
	checkErrors(b, errors)
}

// ipNosimd computes inner product with serial
// addition order of the terms (to make the)
// check comparison turn out right.
func ipNosimd(x, y []float32) float32 {
	var z float32
	for i, a := range x {
		z += a * y[i]
	}
	return z
}

// BenchmarkIPnosimd0 is serial, just a vanilla inner product.
func BenchmarkIPnosimd0(b *testing.B) {
	x := make([]float32, ipBenchLen)
	y := make([]float32, ipBenchLen)

	fill(x, y)

	ip0 := ipNosimd(x, y)

	var errors int
	for b.Loop() {
		var z float32
		for i, a := range x {
			z += a * y[i]
		}
		if z != ip0 {
			errors++
		}
	}
	checkErrors(b, errors)
}

// BenchmarkIPnosimd1 is serial, but with a no-op subslice that
// makes it clear that x and y have the same length.
func BenchmarkIPnosimd1(b *testing.B) {
	x := make([]float32, ipBenchLen)
	y := make([]float32, ipBenchLen)

	fill(x, y)

	ip0 := ipNosimd(x, y)

	var errors int
	for b.Loop() {
		var z float32
		yy := y[:(len(x))]
		for i, a := range x {
			z += a * yy[i]
		}
		if z != ip0 {
			errors++
		}
	}
	checkErrors(b, errors)
}

// BenchmarkIPnosimdA is serial, rewritten to use arrays instead of slices,
// so no bounds checking, gosh darn it to heck.
func BenchmarkIPnosimdA(b *testing.B) {
	var x, y [ipBenchLen]float32

	fill(x[:], y[:])

	ip0 := ipNosimd(x[:], y[:])

	var errors int
	for b.Loop() {
		var z float32
		for i, a := range x {
			z += a * y[i]
		}
		if z != ip0 {
			errors++
		}
	}
	checkErrors(b, errors)
}

var x, y [ipBenchLen]float32
var ip0 float32

func initIp0() {
	fill(x[:], y[:])
	ip0 = ipNosimd(x[:], y[:])
}

// BenchmarkIPnosimdAnotBloop is serial, rewritten to use arrays instead of slices,
// and using a classic iterated loop to see if b.Loop affects subscript inference,
// so no bounds checking, gosh darn it to heck, this time, for sure.
func BenchmarkIPnosimdAnotBloop(b *testing.B) {
	if ip0 == 0 {
		initIp0()
	}

	var errors int
	for range b.N {
		var z float32
		for i, a := range x {
			z += a * y[i]
		}
		if z != ip0 {
			errors++
		}
	}
	checkErrors(b, errors)
}

func ip(x, y []float32) (float32, int, bool) {
	var a simd.Float32s
	sumWidth := a.Len() * 32
	emulated := simd.Emulated()
	var i int
	for i = 0; i < len(x)-a.Len()+1; i += a.Len() {
		u := simd.LoadFloat32s(x[i : i+a.Len()])
		v := simd.LoadFloat32s(y[i : i+a.Len()])
		a = a.Add(u.Mul(v))
	}
	if i < len(x) {
		a = a.Add(first(simd.LoadFloat32sPart(x[i:])).
			Mul(first(simd.LoadFloat32sPart(y[i:]))))
	}

	return sum(a), sumWidth, emulated
}

func ipU(x, y []float32) (float32, int, bool) {
	const U = 4
	var a, a0, a1, a2, a3 simd.Float32s
	sumWidth := a.Len() * 32
	emulated := simd.Emulated()
	var i int
	for i = 0; i < len(x)-U*a.Len()+1; i += U * a.Len() {
		i0 := i
		i1 := i + a.Len()
		i2 := i + 2*a.Len()
		i3 := i + 3*a.Len()

		u := simd.LoadFloat32s(x[i0 : i0+a.Len()])
		v := simd.LoadFloat32s(y[i0 : i0+a.Len()])
		a0 = a0.Add(u.Mul(v))

		u = simd.LoadFloat32s(x[i1 : i1+a.Len()])
		v = simd.LoadFloat32s(y[i1 : i1+a.Len()])
		a1 = a1.Add(u.Mul(v))

		u = simd.LoadFloat32s(x[i2 : i2+a.Len()])
		v = simd.LoadFloat32s(y[i2 : i2+a.Len()])
		a2 = a2.Add(u.Mul(v))

		u = simd.LoadFloat32s(x[i3 : i3+a.Len()])
		v = simd.LoadFloat32s(y[i3 : i3+a.Len()])
		a3 = a3.Add(u.Mul(v))
	}
	a = a0.Add(a1).Add(a2.Add(a3))
	for ; i < len(x)-a.Len()+1; i += a.Len() {
		u := simd.LoadFloat32s(x[i : i+a.Len()])
		v := simd.LoadFloat32s(y[i : i+a.Len()])
		a = a.Add(u.Mul(v))
	}
	if i < len(x) {
		a = a.Add(first(simd.LoadFloat32sPart(x[i:])).
			Mul(first(simd.LoadFloat32sPart(y[i:]))))
	}

	return sum(a), sumWidth, emulated
}

func ipUmore(x, y []float32) (float32, int, bool) {
	const U = 5
	var a, a0, a1, a2, a3, a4 simd.Float32s
	sumWidth := a.Len() * 32
	emulated := simd.Emulated()
	var i int
	for i = 0; i < len(x)-U*a.Len()+1; i += U * a.Len() {
		i0 := i
		i1 := i + a.Len()
		i2 := i + 2*a.Len()
		i3 := i + 3*a.Len()
		i4 := i + 4*a.Len()

		u := simd.LoadFloat32s(x[i0 : i0+a.Len()])
		v := simd.LoadFloat32s(y[i0 : i0+a.Len()])
		a0 = a0.Add(u.Mul(v))

		u = simd.LoadFloat32s(x[i1 : i1+a.Len()])
		v = simd.LoadFloat32s(y[i1 : i1+a.Len()])
		a1 = a1.Add(u.Mul(v))

		u = simd.LoadFloat32s(x[i2 : i2+a.Len()])
		v = simd.LoadFloat32s(y[i2 : i2+a.Len()])
		a2 = a2.Add(u.Mul(v))

		u = simd.LoadFloat32s(x[i3 : i3+a.Len()])
		v = simd.LoadFloat32s(y[i3 : i3+a.Len()])
		a3 = a3.Add(u.Mul(v))

		u = simd.LoadFloat32s(x[i4 : i4+a.Len()])
		v = simd.LoadFloat32s(y[i4 : i4+a.Len()])
		a4 = a4.Add(u.Mul(v))
	}
	a = a0.Add(a1).Add(a2.Add(a3)).Add(a4)

	for ; i < len(x)-a.Len()+1; i += a.Len() {
		u := simd.LoadFloat32s(x[i : i+a.Len()])
		v := simd.LoadFloat32s(y[i : i+a.Len()])
		a = a.Add(u.Mul(v))
	}
	if i < len(x) {
		a = a.Add(first(simd.LoadFloat32sPart(x[i:])).
			Mul(first(simd.LoadFloat32sPart(y[i:]))))
	}

	return sum(a), sumWidth, emulated
}

func ipGoTo(x, y []float32) (float32, int, bool) {
	var a simd.Float32s
	sumWidth := a.Len() * 32
	emulated := simd.Emulated()
	var i int
	var u, v simd.Float32s
loop:
	if !(i < len(x)-a.Len()+1) {
		goto done
	}
	u = simd.LoadFloat32s(x[i : i+a.Len()])
	v = simd.LoadFloat32s(y[i : i+a.Len()])
	a = a.Add(u.Mul(v))
	i += a.Len()
	goto loop
done:
	if i < len(x) {
		a = a.Add(first(simd.LoadFloat32sPart(x[i:])).
			Mul(first(simd.LoadFloat32sPart(y[i:]))))
	}

	return sum(a), sumWidth, emulated
}

func boringSum(x simd.Float32s) float32 {
	s := make([]float32, x.Len())
	x.Store(s)
	var r float32
	for _, e := range s {
		r += e
	}
	return r
}
