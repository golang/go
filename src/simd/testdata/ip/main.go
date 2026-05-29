// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"simd"
)

var sumWidth int
var emulated bool

func main() {
	var a, b [50]float32
	for i := 0; i < 50; i++ {
		a[i] = float32(i)
		b[i] = float32(i)
	}
	fmt.Println(ip(a[:5], b[:5]))
	fmt.Println(ip(a[:10], b[:10]))
	fmt.Println(ip(a[:20], b[:20]))
	fmt.Println(ip(a[:30], b[:30]))
	fmt.Println(ip(a[:40], b[:40]))
	fmt.Println(ip(a[:50], b[:50]))

	fmt.Printf("sum was computed in width %d, emulated = %v\n", sumWidth, emulated)
}

func first[T, U any](t T, u U) T {
	return t
}

func ip(x, y []float32) float32 {
	var a simd.Float32s
	sumWidth = a.Len() * 32
	emulated = simd.Emulated()
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

	return sum(a)
}
