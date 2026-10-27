// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

// SGEMM (C += A*B on float32) written with the scalable SVE API, checked
// against a scalar reference and benchmarked against the hand-written
// assembly kernel in sgemm_sve_arm64.s.

package simd

import (
	"fmt"
	"simd/archsimd"
	"testing"
)

// sgemmSIMD computes C += A(m×k) * B(k×n), for any m, k, n.
// A, B, C are row-major matrixes, e.g:
//
//	A (2×3)		B (3×6)					=>	C (2×6)
//	[ 1 2 3 ]	[ 10 11 12 13 14 15 ]		[ 140 146 152 158 164 170 ]
//	[ 4 5 6 ]	[ 20 21 22 23 24 25 ]		[ 320 335 350 365 380 395 ]
//				[ 30 31 32 33 34 35 ]
//
// Matmul is paralleized at the C-row level, the final C is completed in
// in mxn/Float32s.len() steps (the well-aligned case).
// The code documentation in the function body are written assuming 2D slice,
// i.e. a[i][j] is a[i * m + j].
func sgemmSIMD(c, a, b []float32, m, k, n int) {
	if len(c) < m*n || len(a) < m*k || len(b) < k*n {
		panic("sgemmSIMD: insufficient input slice length")
	}
	vl := archsimd.Float32s{}.Len()
	allTrue := archsimd.Mask32sAllTrue()
	// Computes the the i-th row of c
	for i := 0; i < m; i++ {
		// Take a[i]
		ai := a[i*k : i*k+k]
		// c[i] will be completed, take its reference here
		ci := c[i*n : i*n+n]
		j := 0
		// SIMD loop, parallel level vl, parallelize on c[i]
		for ; j+vl <= n; j += vl {
			// acc is the accumulator for c[i][j:j+vl]
			var acc archsimd.Float32s
			off := j // offset in b
			// Run over each column of a[i]
			for p := range k {
				// av is [a[i][p], a[i][p], ..., a[i][p]]
				av := archsimd.BroadcastFloat32s(ai[p])
				// bv is [b[p][j], b[p][j+1], ..., b[p][j+vl]]
				bv := archsimd.LoadFloat32s(b[off : off+vl])
				// Update c[i]:
				// c[i][j] += a[i][p] * b[p][j]
				// c[i][j+1] += a[i][p] * b[p][j+1]
				// ...
				// c[i][j+vl] += a[i][p] * b[p][j+vl]
				//
				// Note: MulAdd is a predicated-only instruction,
				// to provide the API unpredicated the compiler
				// will insert a synthesized all-true predicate
				// for the instruction. However LICM doesn't work
				// on SIMD operations yet, this all-true predicate
				// will be re-generated for every iteration of this
				// loop. To avoid this, we explicitly create
				// an all-true predicate for the instruction, bypassing
				// the synthesized all-true predicate, i.e. essentially
				// doing LICM manually.
				acc = av.MulAdd(bv, acc).IfElse(allTrue, acc)
				// Move b to next row
				off += n
			}
			acc.Store(ci[j : j+vl])
		}
		sgemmTail(ci, ai, b, j, n, allTrue)
	}
}

// sgemmTail is the masked tail of the inner-most loop of
// sgemmSIMD and sgemmSIMD2, replacing the panicing unmasked loads
// with the masked partial loads.
func sgemmTail(ci, ai, b []float32, j, n int, allTrue archsimd.Mask32s) {
	if j >= n {
		return
	}
	var acc archsimd.Float32s
	off := j
	for p := range ai {
		av := archsimd.BroadcastFloat32s(ai[p])
		bv, _ := archsimd.LoadFloat32sPart(b[off : off+n-j])
		acc = av.MulAdd(bv, acc).IfElse(allTrue, acc)
		off += n
	}
	acc.StorePart(ci[j:])
}

// sgemmSIMD2 manually unrolled the inner-most loop of sgemmSIMD by 2 times,
// which we can save one broadcast out of it.
func sgemmSIMD2(c, a, b []float32, m, k, n int) {
	if len(c) < m*n || len(a) < m*k || len(b) < k*n {
		panic("sgemmSIMD2: insufficient input slice length")
	}
	vl := archsimd.Float32s{}.Len()
	allTrue := archsimd.Mask32sAllTrue()
	for i := 0; i < m; i++ {
		ai := a[i*k : i*k+k]
		ci := c[i*n : i*n+n]
		j := 0
		for ; j+2*vl <= n; j += 2 * vl {
			var acc0, acc1 archsimd.Float32s
			off := j
			for p := range k {
				av := archsimd.BroadcastFloat32s(ai[p])
				bv0 := archsimd.LoadFloat32s(b[off : off+vl])
				acc0 = av.MulAdd(bv0, acc0).IfElse(allTrue, acc0)
				bv1 := archsimd.LoadFloat32s(b[off+vl : off+2*vl])
				acc1 = av.MulAdd(bv1, acc1).IfElse(allTrue, acc1)
				off += n
			}
			acc0.Store(ci[j : j+vl])
			acc1.Store(ci[j+vl : j+2*vl])
		}
		for ; j+vl <= n; j += vl {
			var acc archsimd.Float32s
			off := j
			for p := range k {
				av := archsimd.BroadcastFloat32s(ai[p])
				bv := archsimd.LoadFloat32s(b[off : off+vl])
				acc = av.MulAdd(bv, acc).IfElse(allTrue, acc)
				off += n
			}
			acc.Store(ci[j : j+vl])
		}
		sgemmTail(ci, ai, b, j, n, allTrue)
	}
}

// sgemmAsm is asm-equivalent of sgemmSIMD.
// Unlike sgemmSIMD it has no masked tail, so n must be a
// multiple of the vector length.
func sgemmAsm(c, a, b []float32, m, k, n int) {
	if len(c) < m*n || len(a) < m*k || len(b) < k*n {
		panic("sgemmAsm: insufficient input slice length")
	}
	vl := archsimd.Float32s{}.Len()
	if n%vl != 0 {
		panic("sgemmAsm: n must be a multiple of the vector length")
	}
	for i := 0; i < m; i++ {
		for j := 0; j <= n-vl; j += vl {
			sgemmKernelSVE(&c[i*n+j], &a[i*k], &b[j], int64(k), int64(n)*4)
		}
	}
}

// sgemmScalar is the scalar reference implementation of sgemm.
func sgemmScalar(c, a, b []float32, m, k, n int) {
	if len(c) < m*n || len(a) < m*k || len(b) < k*n {
		panic("sgemmScalar: insufficient input slice length")
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			s := float32(0)
			for p := 0; p < k; p++ {
				s += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = s
		}
	}
}

// sgemmData returns integer-valued inputs for a(mxk), b(kxn) matrixes.
//
// The test data is integer-valued and small enough that every product and sum is
// exact in float32, so the fused and unfused accumulations agree bit for bit in the scalar impl.
func sgemmData(m, k, n int) (a, b []float32) {
	a = make([]float32, m*k)
	b = make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%9) - 4
	}
	for i := range b {
		b[i] = float32(i%7) - 3
	}
	return
}

var sgemmImpls = []struct {
	name string
	f    func(c, a, b []float32, m, k, n int)
}{
	{"simd", sgemmSIMD},
	{"simd2", sgemmSIMD2},
	{"asm", sgemmAsm},
}

func TestSGEMMSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	vl := archsimd.Float32s{}.Len()
	for _, sz := range []struct{ m, k, n int }{
		{7, 37, 64}, // no tail, full alignment to VL
		{7, 37, 61}, // aligned + unaligned, triggers tail
		{3, 5, 2},   // tail only
	} {
		m, k, n := sz.m, sz.k, sz.n
		a, b := sgemmData(m, k, n)
		want := make([]float32, m*n)
		sgemmScalar(want, a, b, m, k, n)
		for _, impl := range sgemmImpls {
			if impl.name == "asm" && n%vl != 0 {
				continue // the assembly kernel has no masked tail
			}
			got := make([]float32, m*n)
			impl.f(got, a, b, m, k, n)
			for i := range got {
				if got[i] != want[i] {
					t.Errorf("%s %dx%dx%d: c[%d,%d] = %v, want %v", impl.name, m, k, n, i/n, i%n, got[i], want[i])
				}
			}
		}
	}
}

func BenchmarkSGEMMSVE(b *testing.B) {
	if !archsimd.ARM64.SVE() {
		b.Skip("no SVE")
	}
	for _, s := range []int{64, 256} {
		m, k, n := s, s, s
		am, bm := sgemmData(m, k, n)
		c := make([]float32, m*n)
		impls := append([]struct {
			name string
			f    func(c, a, b []float32, m, k, n int)
		}{{"scalar", sgemmScalar}}, sgemmImpls...)
		for _, impl := range impls {
			b.Run(fmt.Sprintf("%s/%d", impl.name, s), func(b *testing.B) {
				for range b.N {
					impl.f(c, am, bm, m, k, n)
				}
				flops := 2 * float64(m) * float64(k) * float64(n)
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}
}
