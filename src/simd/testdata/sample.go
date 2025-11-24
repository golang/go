// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"simd"
	"unsafe"
)

func load(s []float64) simd.Float64x4 {
	return simd.LoadFloat64x4((*[4]float64)(s[:4]))
}

type S1 = simd.Float64x4

type S2 simd.Float64x4

func (s S2) Len() int {
	return simd.Float64x4(s).Len()
}

func (s S2) Load(a []float64) S2 {
	return S2(load(a))
}

func (s S2) Store(a *[4]float64) {
	simd.Float64x4(s).Store(a)
}

func (s S2) Add(a S2) S2 {
	return S2(simd.Float64x4(s).Add(simd.Float64x4(a)))
}

func (s S2) Mul(a S2) S2 {
	return S2(simd.Float64x4(s).Mul(simd.Float64x4(a)))
}

type S3 struct {
	simd.Float64x4
}

func ip64_0(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func ip64_1(a, b []float64) float64 {
	var z S1
	sum := z
	var i int
	stride := z.Len()
	for ; i <= len(a)-stride; i += stride {
		va := load(a[i:])
		vb := load(b[i:])
		sum = sum.Add(va.Mul(vb))
	}
	var tmp [4]float64
	sum.Store(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

func ip64_1a(a, b []float64) float64 {
	var z S1
	sum := z
	var i int
	stride := z.Len()
	for ; i <= len(a)-stride; i += stride {
		va := load(a[i:])
		vb := load(b[i:])
		sum = FMA(sum, va, vb)
	}
	var tmp [4]float64
	sum.Store(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

//go:noinline
func FMA(a, b, c simd.Float64x4) simd.Float64x4 {
	return a.Add(b.Mul(c))
}

func ip64_2(a, b []float64) float64 {
	var z S2
	sum := z
	var i int
	stride := z.Len()
	for ; i <= len(a)-stride; i += stride {
		va := z.Load(a[i:])
		vb := z.Load(b[i:])
		sum = sum.Add(va.Mul(vb))
	}
	var tmp [4]float64
	sum.Store(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

func ip64_3(a, b []float64) float64 {
	var z S3
	sum := z
	var i int
	stride := z.Len()
	for ; i <= len(a)-stride; i += stride {
		va := load(a[i:])
		vb := load(b[i:])
		sum = S3{sum.Add(va.Mul(vb))}
	}
	var tmp [4]float64
	sum.Store(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

func main() {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	ip0 := ip64_0(a, a)
	ip1 := ip64_1(a, a)
	ip1a := ip64_1a(a, a)
	ip2 := ip64_2(a, a)
	ip3 := ip64_3(a, a)
	fmt.Printf("Test IP    = %f\n", ip0)
	fmt.Printf("SIMD IP 1  = %f\n", ip1)
	fmt.Printf("SIMD IP 1a = %f\n", ip1a)
	fmt.Printf("SIMD IP 2  = %f\n", ip2)
	fmt.Printf("SIMD IP 3 = %f\n", ip3)
	var z1 S1
	var z2 S2
	var z3 S2

	s1, s2, s3 := unsafe.Sizeof(z1), unsafe.Sizeof(z2), unsafe.Sizeof(z3)

	fmt.Printf("unsafe.Sizeof(z1, z2, z3)=%d, %d, %d\n", s1, s2, s3)

	fail := false

	if s1 != 32 || s2 != 32 || s3 != 32 {
		fmt.Println("Failed a sizeof check, should all be 32")
		fail = true
	}

	if ip1 != ip0 || ip1a != ip0 || ip2 != ip0 || ip3 != ip0 {
		fmt.Println("Failed an inner product check, should all be", ip0)
		fail = true
	}

	if fail {
		os.Exit(1)
	}
}
