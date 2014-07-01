// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Various tests for expressions with high complexity.

package main

// Concatenate 16 4-bit integers into a 64-bit number.
func concat(s *[16]byte) uint64 {
	r := (((((((((((((((uint64(s[0])<<4|
		uint64(s[1]))<<4|
		uint64(s[2]))<<4|
		uint64(s[3]))<<4|
		uint64(s[4]))<<4|
		uint64(s[5]))<<4|
		uint64(s[6]))<<4|
		uint64(s[7]))<<4|
		uint64(s[8]))<<4|
		uint64(s[9]))<<4|
		uint64(s[10]))<<4|
		uint64(s[11]))<<4|
		uint64(s[12]))<<4|
		uint64(s[13]))<<4|
		uint64(s[14]))<<4 |
		uint64(s[15]))
	return r
}

// Compute the determinant of a 4x4-matrix by the sum
// over all index permutations.
func determinant(m [4][4]float64) float64 {
	return m[0][0]*m[1][1]*m[2][2]*m[3][3] -
		m[0][0]*m[1][1]*m[2][3]*m[3][2] -
		m[0][0]*m[1][2]*m[2][1]*m[3][3] +
		m[0][0]*m[1][2]*m[2][3]*m[3][1] +
		m[0][0]*m[1][3]*m[2][1]*m[3][2] -
		m[0][0]*m[1][3]*m[2][2]*m[3][1] -
		m[0][1]*m[1][0]*m[2][2]*m[3][3] +
		m[0][1]*m[1][0]*m[2][3]*m[3][2] +
		m[0][1]*m[1][2]*m[2][0]*m[3][3] -
		m[0][1]*m[1][2]*m[2][3]*m[3][0] -
		m[0][1]*m[1][3]*m[2][0]*m[3][2] +
		m[0][1]*m[1][3]*m[2][2]*m[3][0] +
		m[0][2]*m[1][0]*m[2][1]*m[3][3] -
		m[0][2]*m[1][0]*m[2][3]*m[3][1] -
		m[0][2]*m[1][1]*m[2][0]*m[3][3] +
		m[0][2]*m[1][1]*m[2][3]*m[3][0] +
		m[0][2]*m[1][3]*m[2][0]*m[3][1] -
		m[0][2]*m[1][3]*m[2][1]*m[3][0] -
		m[0][3]*m[1][0]*m[2][1]*m[3][2] +
		m[0][3]*m[1][0]*m[2][2]*m[3][1] +
		m[0][3]*m[1][1]*m[2][0]*m[3][2] -
		m[0][3]*m[1][1]*m[2][2]*m[3][0] -
		m[0][3]*m[1][2]*m[2][0]*m[3][1] +
		m[0][3]*m[1][2]*m[2][1]*m[3][0]
}

// Compute the determinant of a 4x4-matrix by the sum
// over all index permutations.
func determinantInt(m [4][4]int) int {
	return m[0][0]*m[1][1]*m[2][2]*m[3][3] -
		m[0][0]*m[1][1]*m[2][3]*m[3][2] -
		m[0][0]*m[1][2]*m[2][1]*m[3][3] +
		m[0][0]*m[1][2]*m[2][3]*m[3][1] +
		m[0][0]*m[1][3]*m[2][1]*m[3][2] -
		m[0][0]*m[1][3]*m[2][2]*m[3][1] -
		m[0][1]*m[1][0]*m[2][2]*m[3][3] +
		m[0][1]*m[1][0]*m[2][3]*m[3][2] +
		m[0][1]*m[1][2]*m[2][0]*m[3][3] -
		m[0][1]*m[1][2]*m[2][3]*m[3][0] -
		m[0][1]*m[1][3]*m[2][0]*m[3][2] +
		m[0][1]*m[1][3]*m[2][2]*m[3][0] +
		m[0][2]*m[1][0]*m[2][1]*m[3][3] -
		m[0][2]*m[1][0]*m[2][3]*m[3][1] -
		m[0][2]*m[1][1]*m[2][0]*m[3][3] +
		m[0][2]*m[1][1]*m[2][3]*m[3][0] +
		m[0][2]*m[1][3]*m[2][0]*m[3][1] -
		m[0][2]*m[1][3]*m[2][1]*m[3][0] -
		m[0][3]*m[1][0]*m[2][1]*m[3][2] +
		m[0][3]*m[1][0]*m[2][2]*m[3][1] +
		m[0][3]*m[1][1]*m[2][0]*m[3][2] -
		m[0][3]*m[1][1]*m[2][2]*m[3][0] -
		m[0][3]*m[1][2]*m[2][0]*m[3][1] +
		m[0][3]*m[1][2]*m[2][1]*m[3][0]
}

// Compute the determinant of a 4x4-matrix by the sum
// over all index permutations.
func determinantByte(m [4][4]byte) byte {
	return m[0][0]*m[1][1]*m[2][2]*m[3][3] -
		m[0][0]*m[1][1]*m[2][3]*m[3][2] -
		m[0][0]*m[1][2]*m[2][1]*m[3][3] +
		m[0][0]*m[1][2]*m[2][3]*m[3][1] +
		m[0][0]*m[1][3]*m[2][1]*m[3][2] -
		m[0][0]*m[1][3]*m[2][2]*m[3][1] -
		m[0][1]*m[1][0]*m[2][2]*m[3][3] +
		m[0][1]*m[1][0]*m[2][3]*m[3][2] +
		m[0][1]*m[1][2]*m[2][0]*m[3][3] -
		m[0][1]*m[1][2]*m[2][3]*m[3][0] -
		m[0][1]*m[1][3]*m[2][0]*m[3][2] +
		m[0][1]*m[1][3]*m[2][2]*m[3][0] +
		m[0][2]*m[1][0]*m[2][1]*m[3][3] -
		m[0][2]*m[1][0]*m[2][3]*m[3][1] -
		m[0][2]*m[1][1]*m[2][0]*m[3][3] +
		m[0][2]*m[1][1]*m[2][3]*m[3][0] +
		m[0][2]*m[1][3]*m[2][0]*m[3][1] -
		m[0][2]*m[1][3]*m[2][1]*m[3][0] -
		m[0][3]*m[1][0]*m[2][1]*m[3][2] +
		m[0][3]*m[1][0]*m[2][2]*m[3][1] +
		m[0][3]*m[1][1]*m[2][0]*m[3][2] -
		m[0][3]*m[1][1]*m[2][2]*m[3][0] -
		m[0][3]*m[1][2]*m[2][0]*m[3][1] +
		m[0][3]*m[1][2]*m[2][1]*m[3][0]
}

type A []A

// A sequence of constant indexings.
func IndexChain1(s A) A {
	return s[0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0]
}

// A sequence of non-constant indexings.
func IndexChain2(s A, i int) A {
	return s[i][i][i][i][i][i][i][i][i][i][i][i][i][i][i][i]
}

// Another sequence of indexings.
func IndexChain3(s []int) int {
	return s[s[s[s[s[s[s[s[s[s[s[s[s[s[s[s[s[s[s[s[s[0]]]]]]]]]]]]]]]]]]]]]
}

// A right-leaning tree of byte multiplications.
func righttree(a, b, c, d uint8) uint8 {
	return a * (b * (c * (d *
		(a * (b * (c * (d *
			(a * (b * (c * (d *
				(a * (b * (c * (d *
					(a * (b * (c * (d *
						a * (b * (c * d)))))))))))))))))))))

}

// A left-leaning tree of byte multiplications.
func lefttree(a, b, c, d uint8) uint8 {
	return ((((((((((((((((((a * b) * c) * d *
		a) * b) * c) * d *
		a) * b) * c) * d *
		a) * b) * c) * d *
		a) * b) * c) * d *
		a) * b) * c) * d)
}

type T struct {
	Next I
}

type I interface{}

// A chains of type assertions.
func ChainT(t *T) *T {
	return t.
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T).
		Next.(*T)
}

type U struct {
	Children []J
}

func (u *U) Child(n int) J { return u.Children[n] }

type J interface {
	Child(n int) J
}

func ChainUAssert(u *U) *U {
	return u.Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U).
		Child(0).(*U)
}

func ChainUNoAssert(u *U) *U {
	return u.Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).
		Child(0).(*U)
}

// Type assertions and slice indexing. See issue 4207.
func ChainAssertIndex(u *U) J {
	return u.
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0].(*U).
		Children[0]
}

type UArr struct {
	Children [2]J
}

func (u *UArr) Child(n int) J { return u.Children[n] }

func ChainAssertArrayIndex(u *UArr) J {
	return u.
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0].(*UArr).
		Children[0]
}

type UArrPtr struct {
	Children *[2]J
}

func (u *UArrPtr) Child(n int) J { return u.Children[n] }

func ChainAssertArrayptrIndex(u *UArrPtr) J {
	return u.
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0].(*UArrPtr).
		Children[0]
}

// Chains of divisions. See issue 4201.

func ChainDiv(a, b int) int {
	return a / b / a / b / a / b / a / b /
		a / b / a / b / a / b / a / b /
		a / b / a / b / a / b / a / b
}

func ChainDivRight(a, b int) int {
	return a / (b / (a / (b /
		(a / (b / (a / (b /
			(a / (b / (a / (b /
				(a / (b / (a / (b /
					(a / (b / (a / b))))))))))))))))))
}

func ChainDivConst(a int) int {
	return a / 17 / 17 / 17 /
		17 / 17 / 17 / 17 /
		17 / 17 / 17 / 17
}

func ChainMulBytes(a, b, c byte) byte {
	return a*(a*(a*(a*(a*(a*(a*(a*(a*b+c)+c)+c)+c)+c)+c)+c)+c) + c
}

func ChainCap() {
	select {
	case <-make(chan int, cap(make(chan int, cap(make(chan int, cap(make(chan int, cap(make(chan int))))))))):
	default:
	}
}
