// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"fmt"
	"simd"
	"testing"
)

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
