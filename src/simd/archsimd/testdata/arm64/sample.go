// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

// This is a copy of amd64 sample.go adapted for Float32x4 (NEON 128-bit registers).
package main

import (
	"fmt"
	"os"
	"simd/archsimd"
	"unsafe"
)

func load(s []float32) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[4]float32)(s[:4]))
}

type S1 = archsimd.Float32x4

type S2 archsimd.Float32x4

func (s S2) Len() int {
	return archsimd.Float32x4(s).Len()
}

func (s S2) Load(a []float32) S2 {
	return S2(load(a))
}

func (s S2) Store(a *[4]float32) {
	archsimd.Float32x4(s).StoreArray(a)
}

func (s S2) Add(a S2) S2 {
	return S2(archsimd.Float32x4(s).Add(archsimd.Float32x4(a)))
}

func (s S2) Mul(a S2) S2 {
	return S2(archsimd.Float32x4(s).Mul(archsimd.Float32x4(a)))
}

type S3 struct {
	archsimd.Float32x4
}

func ip32_0(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func ip32_1(a, b []float32) float32 {
	var z S1
	sum := z
	var i int
	stride := z.Len()
	for ; i <= len(a)-stride; i += stride {
		va := load(a[i:])
		vb := load(b[i:])
		sum = sum.Add(va.Mul(vb))
	}
	var tmp [4]float32
	sum.StoreArray(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

func ip32_1a(a, b []float32) float32 {
	var z S1
	sum := z
	var i int
	stride := z.Len()
	for ; i <= len(a)-stride; i += stride {
		va := load(a[i:])
		vb := load(b[i:])
		sum = FMA(sum, va, vb)
	}
	var tmp [4]float32
	sum.StoreArray(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

//go:noinline
func FMA(a, b, c archsimd.Float32x4) archsimd.Float32x4 {
	return a.Add(b.Mul(c))
}

func ip32_2(a, b []float32) float32 {
	var z S2
	sum := z
	var i int
	stride := z.Len()
	for ; i <= len(a)-stride; i += stride {
		va := z.Load(a[i:])
		vb := z.Load(b[i:])
		sum = sum.Add(va.Mul(vb))
	}
	var tmp [4]float32
	sum.Store(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

func ip32_3(a, b []float32) float32 {
	var z S3
	sum := z
	var i int
	stride := z.Len()
	for ; i <= len(a)-stride; i += stride {
		va := load(a[i:])
		vb := load(b[i:])
		sum = S3{sum.Add(va.Mul(vb))}
	}
	var tmp [4]float32
	sum.StoreArray(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

func main() {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	ip0 := ip32_0(a, a)
	ip1 := ip32_1(a, a)
	ip1a := ip32_1a(a, a)
	ip2 := ip32_2(a, a)
	ip3 := ip32_3(a, a)
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

	if s1 != 16 || s2 != 16 || s3 != 16 {
		fmt.Println("Failed a sizeof check, should all be 16")
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
