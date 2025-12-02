// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package simd_test

import (
	"fmt"
	"simd"
	"testing"
)

func Transpose4(a0, a1, a2, a3 simd.Int32x4) (b0, b1, b2, b3 simd.Int32x4) {
	t0, t1 := a0.InterleaveLo(a1), a0.InterleaveHi(a1)
	t2, t3 := a2.InterleaveLo(a3), a2.InterleaveHi(a3)

	// a0: ABCD ==> t0: A1B2
	// a1: 1234     t1: C3D4
	// a2: EFGH     t2: E5F6
	// a3: 5678     t3: G7H8

	// need
	// A1E5
	// B2F6
	// C3G7
	// D4H8

	b0 = t0.SelectFromPair(0, 1, 4, 5, t2) // lower elements from each
	b1 = t0.SelectFromPair(2, 3, 6, 7, t2) // upper elements from each
	b2 = t1.SelectFromPair(0, 1, 4, 5, t3) // lowers
	b3 = t1.SelectFromPair(2, 3, 6, 7, t3) // uppers
	return
}

func Transpose8(a0, a1, a2, a3, a4, a5, a6, a7 simd.Int32x8) (b0, b1, b2, b3, b4, b5, b6, b7 simd.Int32x8) {
	t0, t1 := a0.InterleaveLoGrouped(a1), a0.InterleaveHiGrouped(a1)
	t2, t3 := a2.InterleaveLoGrouped(a3), a2.InterleaveHiGrouped(a3)
	t4, t5 := a4.InterleaveLoGrouped(a5), a4.InterleaveHiGrouped(a5)
	t6, t7 := a6.InterleaveLoGrouped(a7), a6.InterleaveHiGrouped(a7)

	// a0: ABCD ==> t0: A1B2
	// a1: 1234     t1: C3D4
	// a2: EFGH     t2: E5F6
	// a3: 5678     t3: G7H8

	// need
	// A1E5
	// B2F6
	// C3G7
	// D4H8

	a0 = t0.SelectFromPairGrouped(0, 1, 4, 5, t2) // lower elements from each
	a1 = t0.SelectFromPairGrouped(2, 3, 6, 7, t2) // upper elements from each
	a2 = t1.SelectFromPairGrouped(0, 1, 4, 5, t3) // lowers
	a3 = t1.SelectFromPairGrouped(2, 3, 6, 7, t3) // uppers

	a4 = t4.SelectFromPairGrouped(0, 1, 4, 5, t6) // lower elements from each
	a5 = t4.SelectFromPairGrouped(2, 3, 6, 7, t6) // upper elements from each
	a6 = t5.SelectFromPairGrouped(0, 1, 4, 5, t7) // lowers
	a7 = t5.SelectFromPairGrouped(2, 3, 6, 7, t7) // uppers

	// next need to swap the upper 128 bits of a0-a3 with the lower 128 bits of a4-a7

	b0 = a0.Select128FromPair(0, 2, a4)
	b4 = a0.Select128FromPair(1, 3, a4)

	b1 = a1.Select128FromPair(0, 2, a5)
	b5 = a1.Select128FromPair(1, 3, a5)

	b2 = a2.Select128FromPair(0, 2, a6)
	b6 = a2.Select128FromPair(1, 3, a6)

	b3 = a3.Select128FromPair(0, 2, a7)
	b7 = a3.Select128FromPair(1, 3, a7)

	return
}

func TestTranspose4(t *testing.T) {
	r := make([]int32, 16, 16)

	w := simd.LoadInt32x4Slice([]int32{0xA, 0xB, 0xC, 0xD})
	x := simd.LoadInt32x4Slice([]int32{1, 2, 3, 4})
	y := simd.LoadInt32x4Slice([]int32{0xE, 0xF, 0x10, 0x11})
	z := simd.LoadInt32x4Slice([]int32{5, 6, 7, 8})
	a, b, c, d := Transpose4(w, x, y, z)

	a.StoreSlice(r[0:])
	b.StoreSlice(r[4:])
	c.StoreSlice(r[8:])
	d.StoreSlice(r[12:])

	checkSlices[int32](t, r, []int32{
		0xA, 1, 0xE, 5,
		0xB, 2, 0xF, 6,
		0xC, 3, 0x10, 7,
		0xD, 4, 0x11, 8,
	})

}

func TestTranspose8(t *testing.T) {
	m := make([]int32, 8)

	a := []int32{}
	for i := int32(1); i <= 64; i++ {
		a = append(a, i)
	}

	p := simd.LoadInt32x8Slice(a[0:])
	q := simd.LoadInt32x8Slice(a[8:])
	r := simd.LoadInt32x8Slice(a[16:])
	s := simd.LoadInt32x8Slice(a[24:])

	w := simd.LoadInt32x8Slice(a[32:])
	x := simd.LoadInt32x8Slice(a[40:])
	y := simd.LoadInt32x8Slice(a[48:])
	z := simd.LoadInt32x8Slice(a[56:])

	p, q, r, s, w, x, y, z = Transpose8(p, q, r, s, w, x, y, z)

	foo := func(a simd.Int32x8, z int32) {
		a.StoreSlice(m)
		var o []int32
		for i := int32(0); i < 8; i++ {
			o = append(o, z+i*8)
		}
		checkSlices[int32](t, m, o)
	}

	foo(p, 1)
	foo(q, 2)
	foo(r, 3)
	foo(s, 4)
	foo(w, 5)
	foo(x, 6)
	foo(y, 7)
	foo(z, 8)

}

const BIG = 20000

var bigMatrix [][]int32

// 9x9 is smallest matrix with diagonal and off-diagonal tiles, plus a fringe.
var nineMatrix [][]int32

var thirtyMatrix [][]int32

func fill(m [][]int32) {
	for i := range m {
		m[i] = make([]int32, len(m))
		for j := range m[i] {
			m[i][j] = int32(-i<<16 + j)
		}
	}
}

func isTransposed(m [][]int32) bool {
	for i, mi := range m {
		for j, a := range mi {
			if a != int32(-j<<16+i) {
				return false
			}
		}
	}
	return true
}

func dupe(m [][]int32) [][]int32 {
	n := len(m)
	p := make([][]int32, n, n)
	for i := range p {
		t := make([]int32, n)
		for j, a := range m[i] {
			t[j] = a
		}
		p[i] = t
	}
	return p
}

func init() {
	bigMatrix = make([][]int32, BIG, BIG)
	fill(bigMatrix)
	nineMatrix = make([][]int32, 9, 9)
	fill(nineMatrix)
	thirtyMatrix = make([][]int32, 30, 30)
	fill(thirtyMatrix)
}

func BenchmarkPlainTranspose(b *testing.B) {
	d := dupe(bigMatrix)
	for b.Loop() {
		transposePlain(d)
	}
}

func BenchmarkTiled4Transpose(b *testing.B) {
	d := dupe(bigMatrix)
	for b.Loop() {
		transposeTiled4(d)
	}
}

func BenchmarkTiled8Transpose(b *testing.B) {
	d := dupe(bigMatrix)
	for b.Loop() {
		transposeTiled8(d)
	}
}

func Benchmark2BlockedTranspose(b *testing.B) {
	d := dupe(bigMatrix)
	for b.Loop() {
		transpose2Blocked(d)
	}
}
func Benchmark3BlockedTranspose(b *testing.B) {
	d := dupe(bigMatrix)
	for b.Loop() {
		transpose3Blocked(d)
	}
}
func Benchmark4BlockedTranspose(b *testing.B) {
	d := dupe(bigMatrix)
	for b.Loop() {
		transpose4Blocked(d)
	}
}
func Benchmark5aBlockedTranspose(b *testing.B) {
	d := dupe(bigMatrix)
	for b.Loop() {
		transpose5aBlocked(d)
	}
}

func Benchmark5bBlockedTranspose(b *testing.B) {
	d := dupe(bigMatrix)
	for b.Loop() {
		transpose5bBlocked(d)
	}
}

func transposePlain(m [][]int32) {
	for i := range m {
		for j := 0; j < i; j++ {
			t := m[i][j]
			m[i][j] = m[j][i]
			m[j][i] = t
		}
	}
}

func TestTransposePlain(t *testing.T) {
	d := dupe(nineMatrix)
	t.Logf("Input matrix is %s", formatMatrix(d))
	transposePlain(d)
	if !isTransposed(d) {
		t.Errorf("d is not transposed, d = %s", formatMatrix(d))
	} else {
		t.Logf("Transposed plain matrix = %s", formatMatrix(d))
	}
}

func TestTranspose2Blocked(t *testing.T) {
	d := dupe(nineMatrix)
	t.Logf("Input matrix is %s", formatMatrix(d))
	transpose2Blocked(d)
	if !isTransposed(d) {
		t.Errorf("d is not transposed, d = %s", formatMatrix(d))
	}
}

func TestTranspose3Blocked(t *testing.T) {
	d := dupe(nineMatrix)
	t.Logf("Input matrix is %s", formatMatrix(d))
	transpose3Blocked(d)
	if !isTransposed(d) {
		t.Errorf("d is not transposed, d = %s", formatMatrix(d))
	}
}

func TestTranspose4Blocked(t *testing.T) {
	d := dupe(nineMatrix)
	t.Logf("Input matrix is %s", formatMatrix(d))
	transpose4Blocked(d)
	if !isTransposed(d) {
		t.Errorf("d is not transposed, d = %s", formatMatrix(d))
	}
}

func TestTranspose5aBlocked(t *testing.T) {
	d := dupe(nineMatrix)
	t.Logf("Input matrix is %s", formatMatrix(d))
	transpose5aBlocked(d)
	if !isTransposed(d) {
		t.Errorf("d is not transposed, d = %s", formatMatrix(d))
	}
}

func TestTranspose5bBlocked(t *testing.T) {
	d := dupe(nineMatrix)
	t.Logf("Input matrix is %s", formatMatrix(d))
	transpose5bBlocked(d)
	if !isTransposed(d) {
		t.Errorf("d is not transposed, d = %s", formatMatrix(d))
	}
}

func TestTransposeTiled4(t *testing.T) {
	d := dupe(nineMatrix)
	transposeTiled4(d)
	if !isTransposed(d) {
		t.Errorf("d is not transposed, d = %v", d)
	}
}

func TestTransposeTiled8(t *testing.T) {
	d := dupe(thirtyMatrix)
	transposeTiled8(d)
	if !isTransposed(d) {
		t.Errorf("d is not transposed, d = %v", d)
	}
}

func formatMatrix(m [][]int32) string {
	s := ""
	for _, mi := range m {
		s += "\n["
		for _, t := range mi {
			h := t >> 16
			l := t & 0xffff
			s += fmt.Sprintf(" (%d %d)", h, l)
		}
		s += " ]"
	}
	return s
}

func transpose2Blocked(m [][]int32) {
	const B = 2
	N := len(m)
	i := 0
	for ; i <= len(m)-B; i += B {
		r0, r1 := m[i], m[i+1]
		if len(r0) < N || len(r1) < N {
			panic("Early bounds check failure")
		}
		// transpose around diagonal
		d01, d10 := r0[i+1], r1[i]
		r0[i+1], r1[i] = d10, d01

		// transpose across diagonal
		j := 0
		for ; j < i; j += B {
			a0, a1 := m[j], m[j+1]

			b00, b01 := a0[i], a0[i+1]
			b10, b11 := a1[i], a1[i+1]

			a0[i], a0[i+1] = r0[j], r1[j]
			a1[i], a1[i+1] = r0[j+1], r1[j+1]

			r0[j], r0[j+1] = b00, b10
			r1[j], r1[j+1] = b01, b11
		}
	}

	// Do the fringe
	for ; i < len(m); i++ {
		j := 0
		r := m[i]
		for ; j < i; j++ {
			t := r[j]
			r[j] = m[j][i]
			m[j][i] = t
		}
	}
}

func transpose3Blocked(m [][]int32) {
	const B = 3
	N := len(m)
	i := 0
	for ; i <= len(m)-B; i += B {
		r0, r1, r2 := m[i], m[i+1], m[i+2]
		if len(r0) < N || len(r1) < N {
			panic("Early bounds check failure")
		}
		// transpose around diagonal
		d01, d10 := r0[i+1], r1[i]
		d02, d20 := r0[i+2], r2[i]
		d12, d21 := r1[i+2], r2[i+1]

		r0[i+1], r1[i] = d10, d01
		r0[i+2], r2[i] = d20, d02
		r1[i+2], r2[i+1] = d21, d12

		// transpose across diagonal
		j := 0
		for ; j < i; j += B {
			a0, a1, a2 := m[j], m[j+1], m[j+2]

			b00, b01, b02 := a0[i], a0[i+1], a0[i+2]
			b10, b11, b12 := a1[i], a1[i+1], a1[i+2]
			b20, b21, b22 := a2[i], a2[i+1], a2[i+2]

			a0[i], a0[i+1], a0[i+2] = r0[j], r1[j], r2[j]
			a1[i], a1[i+1], a1[i+2] = r0[j+1], r1[j+1], r2[j+1]
			a2[i], a2[i+1], a2[i+2] = r0[j+2], r1[j+2], r2[j+2]

			r0[j], r0[j+1], r0[j+2] = b00, b10, b20
			r1[j], r1[j+1], r1[j+2] = b01, b11, b21
			r2[j], r2[j+1], r2[j+2] = b02, b12, b22
		}
	}

	// Do the fringe
	for ; i < len(m); i++ {
		j := 0
		r := m[i]
		for ; j < i; j++ {
			t := r[j]
			r[j] = m[j][i]
			m[j][i] = t
		}
	}
}

func transpose4Blocked(m [][]int32) {
	const B = 4
	N := len(m)
	i := 0
	for ; i <= len(m)-B; i += B {
		r0, r1, r2, r3 := m[i], m[i+1], m[i+2], m[i+3]
		if len(r0) < N || len(r1) < N || len(r2) < N || len(r3) < N {
			panic("Early bounds check failure")
		}
		// transpose around diagonal
		d01, d10 := r0[i+1], r1[i]
		d02, d20 := r0[i+2], r2[i]
		d03, d30 := r0[i+3], r3[i]
		d12, d21 := r1[i+2], r2[i+1]
		d13, d31 := r1[i+3], r3[i+1]
		d23, d32 := r2[i+3], r3[i+2]

		r0[i+1], r1[i] = d10, d01
		r0[i+2], r2[i] = d20, d02
		r0[i+3], r3[i] = d30, d03
		r1[i+2], r2[i+1] = d21, d12
		r1[i+3], r3[i+1] = d31, d13
		r2[i+3], r3[i+2] = d32, d23

		// transpose across diagonal
		j := 0
		for ; j < i; j += B {
			a0, a1, a2, a3 := m[j], m[j+1], m[j+2], m[j+3]

			b00, b01, b02, b03 := a0[i], a0[i+1], a0[i+2], a0[i+3]
			b10, b11, b12, b13 := a1[i], a1[i+1], a1[i+2], a1[i+3]
			b20, b21, b22, b23 := a2[i], a2[i+1], a2[i+2], a2[i+3]
			b30, b31, b32, b33 := a3[i], a3[i+1], a3[i+2], a3[i+3]

			a0[i], a0[i+1], a0[i+2], a0[i+3] = r0[j], r1[j], r2[j], r3[j]
			a1[i], a1[i+1], a1[i+2], a1[i+3] = r0[j+1], r1[j+1], r2[j+1], r3[j+1]
			a2[i], a2[i+1], a2[i+2], a2[i+3] = r0[j+2], r1[j+2], r2[j+2], r3[j+2]
			a3[i], a3[i+1], a3[i+2], a3[i+3] = r0[j+3], r1[j+3], r2[j+3], r3[j+3]

			r0[j], r0[j+1], r0[j+2], r0[j+3] = b00, b10, b20, b30
			r1[j], r1[j+1], r1[j+2], r1[j+3] = b01, b11, b21, b31
			r2[j], r2[j+1], r2[j+2], r2[j+3] = b02, b12, b22, b32
			r3[j], r3[j+1], r3[j+2], r3[j+3] = b03, b13, b23, b33
		}
	}

	// Do the fringe
	for ; i < len(m); i++ {
		j := 0
		r := m[i]
		for ; j < i; j++ {
			t := r[j]
			r[j] = m[j][i]
			m[j][i] = t
		}
	}
}

func transpose5aBlocked(m [][]int32) {
	const B = 5
	N := len(m)
	i := 0
	for ; i <= len(m)-B; i += B {
		r0, r1, r2, r3, r4 := m[i], m[i+1], m[i+2], m[i+3], m[i+4]
		if len(r0) < N || len(r1) < N || len(r2) < N || len(r3) < N || len(r4) < N {
			panic("Early bounds check failure")
		}
		// transpose around diagonal
		d01, d10 := r0[i+1], r1[i]
		d02, d20 := r0[i+2], r2[i]
		d03, d30 := r0[i+3], r3[i]
		d04, d40 := r0[i+4], r4[i]

		d12, d21 := r1[i+2], r2[i+1]
		d13, d31 := r1[i+3], r3[i+1]
		d14, d41 := r1[i+4], r4[i+1]

		d23, d32 := r2[i+3], r3[i+2]
		d24, d42 := r2[i+4], r4[i+2]

		d34, d43 := r3[i+4], r4[i+3]

		r0[i+1], r1[i] = d10, d01
		r0[i+2], r2[i] = d20, d02
		r0[i+3], r3[i] = d30, d03
		r0[i+4], r4[i] = d40, d04

		r1[i+2], r2[i+1] = d21, d12
		r1[i+3], r3[i+1] = d31, d13
		r1[i+4], r4[i+1] = d41, d14

		r2[i+3], r3[i+2] = d32, d23
		r2[i+4], r4[i+2] = d42, d24

		r3[i+4], r4[i+3] = d43, d34

		// transpose across diagonal
		j := 0
		for ; j < i; j += B {
			a0, a1, a2, a3, a4 := m[j], m[j+1], m[j+2], m[j+3], m[j+4]

			b00, b01, b02, b03, b04 := a0[i], a0[i+1], a0[i+2], a0[i+3], a0[i+4]
			b10, b11, b12, b13, b14 := a1[i], a1[i+1], a1[i+2], a1[i+3], a1[i+4]
			b20, b21, b22, b23, b24 := a2[i], a2[i+1], a2[i+2], a2[i+3], a2[i+4]
			b30, b31, b32, b33, b34 := a3[i], a3[i+1], a3[i+2], a3[i+3], a3[i+4]
			b40, b41, b42, b43, b44 := a4[i], a4[i+1], a4[i+2], a4[i+3], a4[i+4]

			a0[i], a0[i+1], a0[i+2], a0[i+3], a0[i+4] = r0[j], r1[j], r2[j], r3[j], r4[j]
			a1[i], a1[i+1], a1[i+2], a1[i+3], a1[i+4] = r0[j+1], r1[j+1], r2[j+1], r3[j+1], r4[j+1]
			a2[i], a2[i+1], a2[i+2], a2[i+3], a2[i+4] = r0[j+2], r1[j+2], r2[j+2], r3[j+2], r4[j+2]
			a3[i], a3[i+1], a3[i+2], a3[i+3], a3[i+4] = r0[j+3], r1[j+3], r2[j+3], r3[j+3], r4[j+3]
			a4[i], a4[i+1], a4[i+2], a4[i+3], a4[i+4] = r0[j+4], r1[j+4], r2[j+4], r3[j+4], r4[j+4]

			r0[j], r0[j+1], r0[j+2], r0[j+3], r0[j+4] = b00, b10, b20, b30, b40
			r1[j], r1[j+1], r1[j+2], r1[j+3], r1[j+4] = b01, b11, b21, b31, b41
			r2[j], r2[j+1], r2[j+2], r2[j+3], r2[j+4] = b02, b12, b22, b32, b42
			r3[j], r3[j+1], r3[j+2], r3[j+3], r3[j+4] = b03, b13, b23, b33, b43
			r4[j], r4[j+1], r4[j+2], r4[j+3], r4[j+4] = b04, b14, b24, b34, b44
		}
	}

	// Do the fringe
	for ; i < len(m); i++ {
		j := 0
		r := m[i]
		for ; j < i; j++ {
			t := r[j]
			r[j] = m[j][i]
			m[j][i] = t
		}
	}
}

// transpose5bBlocked is just like transpose5aBlocked
// but rewritten to reduce register pressure in the
// inner loop.
func transpose5bBlocked(m [][]int32) {
	const B = 5
	N := len(m)
	i := 0
	for ; i <= len(m)-B; i += B {
		r0, r1, r2, r3, r4 := m[i], m[i+1], m[i+2], m[i+3], m[i+4]
		if len(r0) < N || len(r1) < N || len(r2) < N || len(r3) < N || len(r4) < N {
			panic("Early bounds check failure")
		}
		// transpose around diagonal
		d01, d10 := r0[i+1], r1[i]
		d02, d20 := r0[i+2], r2[i]
		d03, d30 := r0[i+3], r3[i]
		d04, d40 := r0[i+4], r4[i]
		r0[i+1], r1[i] = d10, d01
		r0[i+2], r2[i] = d20, d02
		r0[i+3], r3[i] = d30, d03
		r0[i+4], r4[i] = d40, d04

		d12, d21 := r1[i+2], r2[i+1]
		d13, d31 := r1[i+3], r3[i+1]
		d14, d41 := r1[i+4], r4[i+1]
		r1[i+2], r2[i+1] = d21, d12
		r1[i+3], r3[i+1] = d31, d13
		r1[i+4], r4[i+1] = d41, d14

		d23, d32 := r2[i+3], r3[i+2]
		d24, d42 := r2[i+4], r4[i+2]
		r2[i+3], r3[i+2] = d32, d23
		r2[i+4], r4[i+2] = d42, d24

		d34, d43 := r3[i+4], r4[i+3]
		r3[i+4], r4[i+3] = d43, d34

		// transpose across diagonal
		j := 0
		for ; j < i; j += B {
			a4, a0, a1, a2, a3 := m[j+4], m[j], m[j+1], m[j+2], m[j+3]

			// Process column i+4
			temp0 := a0[i+4]
			temp1 := a1[i+4]
			temp2 := a2[i+4]
			temp3 := a3[i+4]
			temp4 := a4[i+4]

			a4[i+4] = r4[j+4]
			a0[i+4] = r4[j]
			a1[i+4] = r4[j+1]
			a2[i+4] = r4[j+2]
			a3[i+4] = r4[j+3]

			r0[j+4] = temp0
			r1[j+4] = temp1
			r2[j+4] = temp2
			r3[j+4] = temp3
			r4[j+4] = temp4

			// Process column i
			temp0 = a0[i]
			temp1 = a1[i]
			temp2 = a2[i]
			temp3 = a3[i]
			temp4 = a4[i]

			a4[i] = r0[j+4]
			a0[i] = r0[j]
			a1[i] = r0[j+1]
			a2[i] = r0[j+2]
			a3[i] = r0[j+3]

			r0[j] = temp0
			r1[j] = temp1
			r2[j] = temp2
			r3[j] = temp3
			r4[j] = temp4

			// Process column i+1
			temp0 = a0[i+1]
			temp1 = a1[i+1]
			temp2 = a2[i+1]
			temp3 = a3[i+1]
			temp4 = a4[i+1]

			a4[i+1] = r1[j+4]
			a0[i+1] = r1[j]
			a1[i+1] = r1[j+1]
			a2[i+1] = r1[j+2]
			a3[i+1] = r1[j+3]

			r0[j+1] = temp0
			r1[j+1] = temp1
			r2[j+1] = temp2
			r3[j+1] = temp3
			r4[j+1] = temp4

			// Process column i+2
			temp0 = a0[i+2]
			temp1 = a1[i+2]
			temp2 = a2[i+2]
			temp3 = a3[i+2]
			temp4 = a4[i+2]

			a4[i+2] = r2[j+4]
			a0[i+2] = r2[j]
			a1[i+2] = r2[j+1]
			a2[i+2] = r2[j+2]
			a3[i+2] = r2[j+3]

			r0[j+2] = temp0
			r1[j+2] = temp1
			r2[j+2] = temp2
			r3[j+2] = temp3
			r4[j+2] = temp4

			// Process column i+3
			temp0 = a0[i+3]
			temp1 = a1[i+3]
			temp2 = a2[i+3]
			temp3 = a3[i+3]
			temp4 = a4[i+3]

			a4[i+3] = r3[j+4]
			a0[i+3] = r3[j]
			a1[i+3] = r3[j+1]
			a2[i+3] = r3[j+2]
			a3[i+3] = r3[j+3]

			r0[j+3] = temp0
			r1[j+3] = temp1
			r2[j+3] = temp2
			r3[j+3] = temp3
			r4[j+3] = temp4
		}
	}

	// Do the fringe
	for ; i < len(m); i++ {
		j := 0
		r := m[i]
		for ; j < i; j++ {
			t := r[j]
			r[j] = m[j][i]
			m[j][i] = t
		}
	}
}

func transposeTiled4(m [][]int32) {
	const B = 4
	N := len(m)
	i := 0
	for ; i < len(m)-(B-1); i += B {
		r0, r1, r2, r3 := m[i], m[i+1], m[i+2], m[i+3]
		if len(r0) < N || len(r1) < N || len(r2) < N || len(r3) < N {
			panic("Early bounds check failure")
		}
		// transpose diagonal
		d0, d1, d2, d3 :=
			simd.LoadInt32x4Slice(r0[i:]),
			simd.LoadInt32x4Slice(r1[i:]),
			simd.LoadInt32x4Slice(r2[i:]),
			simd.LoadInt32x4Slice(r3[i:])

		d0, d1, d2, d3 = Transpose4(d0, d1, d2, d3)

		d0.StoreSlice(r0[i:])
		d1.StoreSlice(r1[i:])
		d2.StoreSlice(r2[i:])
		d3.StoreSlice(r3[i:])

		// transpose across diagonal
		j := 0
		for ; j < i; j += B {
			a0, a1, a2, a3 := m[j], m[j+1], m[j+2], m[j+3]
			u0, u1, u2, u3 :=
				simd.LoadInt32x4Slice(a0[i:]),
				simd.LoadInt32x4Slice(a1[i:]),
				simd.LoadInt32x4Slice(a2[i:]),
				simd.LoadInt32x4Slice(a3[i:])

			u0, u1, u2, u3 = Transpose4(u0, u1, u2, u3)

			l0 := simd.LoadInt32x4Slice(r0[j:])
			u0.StoreSlice(r0[j:])
			l1 := simd.LoadInt32x4Slice(r1[j:])
			u1.StoreSlice(r1[j:])
			l2 := simd.LoadInt32x4Slice(r2[j:])
			u2.StoreSlice(r2[j:])
			l3 := simd.LoadInt32x4Slice(r3[j:])
			u3.StoreSlice(r3[j:])

			u0, u1, u2, u3 = Transpose4(l0, l1, l2, l3)

			u0.StoreSlice(a0[i:])
			u1.StoreSlice(a1[i:])
			u2.StoreSlice(a2[i:])
			u3.StoreSlice(a3[i:])
		}
	}
	// Do the fringe
	for ; i < len(m); i++ {
		j := 0
		r := m[i]
		for ; j < i; j++ {
			t := r[j]
			r[j] = m[j][i]
			m[j][i] = t
		}
	}
}

func transposeTiled8(m [][]int32) {
	const B = 8
	N := len(m)
	i := 0
	for ; i < len(m)-(B-1); i += B {
		r0, r1, r2, r3, r4, r5, r6, r7 := m[i], m[i+1], m[i+2], m[i+3], m[i+4], m[i+5], m[i+6], m[i+7]
		if len(r0) < N || len(r1) < N || len(r2) < N || len(r3) < N || len(r4) < N || len(r5) < N || len(r6) < N || len(r7) < N {
			panic("Early bounds check failure")
		}
		// transpose diagonal
		d0, d1, d2, d3, d4, d5, d6, d7 :=
			simd.LoadInt32x8Slice(r0[i:]),
			simd.LoadInt32x8Slice(r1[i:]),
			simd.LoadInt32x8Slice(r2[i:]),
			simd.LoadInt32x8Slice(r3[i:]),
			simd.LoadInt32x8Slice(r4[i:]),
			simd.LoadInt32x8Slice(r5[i:]),
			simd.LoadInt32x8Slice(r6[i:]),
			simd.LoadInt32x8Slice(r7[i:])

		d0, d1, d2, d3, d4, d5, d6, d7 = Transpose8(d0, d1, d2, d3, d4, d5, d6, d7)

		d0.StoreSlice(r0[i:])
		d1.StoreSlice(r1[i:])
		d2.StoreSlice(r2[i:])
		d3.StoreSlice(r3[i:])
		d4.StoreSlice(r4[i:])
		d5.StoreSlice(r5[i:])
		d6.StoreSlice(r6[i:])
		d7.StoreSlice(r7[i:])

		// transpose across diagonal
		j := 0
		for ; j < i; j += B {
			a7, a0, a1, a2, a3, a4, a5, a6 := m[j+7], m[j], m[j+1], m[j+2], m[j+3], m[j+4], m[j+5], m[j+6]
			u0, u1, u2, u3, u4, u5, u6, u7 :=
				simd.LoadInt32x8Slice(a0[i:]),
				simd.LoadInt32x8Slice(a1[i:]),
				simd.LoadInt32x8Slice(a2[i:]),
				simd.LoadInt32x8Slice(a3[i:]),
				simd.LoadInt32x8Slice(a4[i:]),
				simd.LoadInt32x8Slice(a5[i:]),
				simd.LoadInt32x8Slice(a6[i:]),
				simd.LoadInt32x8Slice(a7[i:])

			u0, u1, u2, u3, u4, u5, u6, u7 = Transpose8(u0, u1, u2, u3, u4, u5, u6, u7)

			l0 := simd.LoadInt32x8Slice(r0[j:])
			u0.StoreSlice(r0[j:])
			l1 := simd.LoadInt32x8Slice(r1[j:])
			u1.StoreSlice(r1[j:])
			l2 := simd.LoadInt32x8Slice(r2[j:])
			u2.StoreSlice(r2[j:])
			l3 := simd.LoadInt32x8Slice(r3[j:])
			u3.StoreSlice(r3[j:])
			l4 := simd.LoadInt32x8Slice(r4[j:])
			u4.StoreSlice(r4[j:])
			l5 := simd.LoadInt32x8Slice(r5[j:])
			u5.StoreSlice(r5[j:])
			l6 := simd.LoadInt32x8Slice(r6[j:])
			u6.StoreSlice(r6[j:])
			l7 := simd.LoadInt32x8Slice(r7[j:])
			u7.StoreSlice(r7[j:])

			u0, u1, u2, u3, u4, u5, u6, u7 = Transpose8(l0, l1, l2, l3, l4, l5, l6, l7)

			u0.StoreSlice(a0[i:])
			u1.StoreSlice(a1[i:])
			u2.StoreSlice(a2[i:])
			u3.StoreSlice(a3[i:])
			u4.StoreSlice(a4[i:])
			u5.StoreSlice(a5[i:])
			u6.StoreSlice(a6[i:])
			u7.StoreSlice(a7[i:])
		}
	}
	// Do the fringe
	for ; i < len(m); i++ {
		j := 0
		r := m[i]
		for ; j < i; j++ {
			t := r[j]
			r[j] = m[j][i]
			m[j][i] = t
		}
	}
}
