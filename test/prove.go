// errorcheck -0 -d=ssa/prove/debug=1

//go:build amd64 || arm64

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/bits"
)

func f0(a []int) int {
	a[0] = 1
	a[0] = 1 // ERROR "Proved IsInBounds$"
	a[6] = 1
	a[6] = 1 // ERROR "Proved IsInBounds$"
	a[5] = 1 // ERROR "Proved IsInBounds$"
	a[5] = 1 // ERROR "Proved IsInBounds$"
	return 13
}

func f1(a []int) int {
	if len(a) <= 5 {
		return 18
	}
	a[0] = 1 // ERROR "Proved IsInBounds$"
	a[0] = 1 // ERROR "Proved IsInBounds$"
	a[6] = 1
	a[6] = 1 // ERROR "Proved IsInBounds$"
	a[5] = 1 // ERROR "Proved IsInBounds$"
	a[5] = 1 // ERROR "Proved IsInBounds$"
	return 26
}

func f1b(a []int, i int, j uint) int {
	if i >= 0 && i < len(a) {
		return a[i] // ERROR "Proved IsInBounds$"
	}
	if i >= 10 && i < len(a) {
		return a[i] // ERROR "Proved IsInBounds$"
	}
	if i >= 10 && i < len(a) {
		return a[i] // ERROR "Proved IsInBounds$"
	}
	if i >= 10 && i < len(a) {
		return a[i-10] // ERROR "Proved IsInBounds$"
	}
	if j < uint(len(a)) {
		return a[j] // ERROR "Proved IsInBounds$"
	}
	return 0
}

func f1c(a []int, i int64) int {
	c := uint64(math.MaxInt64 + 10) // overflows int
	d := int64(c)
	if i >= d && i < int64(len(a)) {
		// d overflows, should not be handled.
		return a[i]
	}
	return 0
}

func f2(a []int) int {
	for i := range a { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		a[i+1] = i
		a[i+1] = i // ERROR "Proved IsInBounds$"
	}
	return 34
}

func f3(a []uint) int {
	for i := uint(0); i < uint(len(a)); i++ {
		a[i] = i // ERROR "Proved IsInBounds$"
	}
	return 41
}

func f4a(a, b, c int) int {
	if a < b {
		if a == b { // ERROR "Disproved Eq64$"
			return 47
		}
		if a > b { // ERROR "Disproved Less64$"
			return 50
		}
		if a < b { // ERROR "Proved Less64$"
			return 53
		}
		// We can't get to this point and prove knows that, so
		// there's no message for the next (obvious) branch.
		if a != a {
			return 56
		}
		return 61
	}
	return 63
}

func f4b(a, b, c int) int {
	if a <= b {
		if a >= b {
			if a == b { // ERROR "Proved Eq64$"
				return 70
			}
			return 75
		}
		return 77
	}
	return 79
}

func f4c(a, b, c int) int {
	if a <= b {
		if a >= b {
			if a != b { // ERROR "Disproved Neq64$"
				return 73
			}
			return 75
		}
		return 77
	}
	return 79
}

func f4d(a, b, c int) int {
	if a < b {
		if a < c {
			if a < b { // ERROR "Proved Less64$"
				if a < c { // ERROR "Proved Less64$"
					return 87
				}
				return 89
			}
			return 91
		}
		return 93
	}
	return 95
}

func f4e(a, b, c int) int {
	if a < b {
		if b > a { // ERROR "Proved Less64$"
			return 101
		}
		return 103
	}
	return 105
}

func f4f(a, b, c int) int {
	if a <= b {
		if b > a {
			if b == a { // ERROR "Disproved Eq64$"
				return 112
			}
			return 114
		}
		if b >= a { // ERROR "Proved Leq64$"
			if b == a { // ERROR "Proved Eq64$"
				return 118
			}
			return 120
		}
		return 122
	}
	return 124
}

func f5(a, b uint) int {
	if a == b {
		if a <= b { // ERROR "Proved Leq64U$"
			return 130
		}
		return 132
	}
	return 134
}

// These comparisons are compile time constants.
func f6a(a uint8) int {
	if a < a { // ERROR "Disproved Less8U$"
		return 140
	}
	return 151
}

func f6b(a uint8) int {
	if a < a { // ERROR "Disproved Less8U$"
		return 140
	}
	return 151
}

func f6x(a uint8) int {
	if a > a { // ERROR "Disproved Less8U$"
		return 143
	}
	return 151
}

func f6d(a uint8) int {
	if a <= a { // ERROR "Proved Leq8U$"
		return 146
	}
	return 151
}

func f6e(a uint8) int {
	if a >= a { // ERROR "Proved Leq8U$"
		return 149
	}
	return 151
}

func f7(a []int, b int) int {
	if b < len(a) {
		a[b] = 3
		if b < len(a) { // ERROR "Proved Less64$"
			a[b] = 5 // ERROR "Proved IsInBounds$"
		}
	}
	return 161
}

func f8(a, b uint) int {
	if a == b {
		return 166
	}
	if a > b {
		return 169
	}
	if a < b { // ERROR "Proved Less64U$"
		return 172
	}
	return 174
}

func f9(a, b bool) int {
	if a {
		return 1
	}
	if a || b { // ERROR "Disproved Arg$"
		return 2
	}
	return 3
}

func f10(a string) int {
	n := len(a)
	// We optimize comparisons with small constant strings (see cmd/compile/internal/gc/walk.go),
	// so this string literal must be long.
	if a[:n>>1] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" {
		return 0
	}
	return 1
}

func f11a(a []int, i int) {
	useInt(a[i])
	useInt(a[i]) // ERROR "Proved IsInBounds$"
}

func f11b(a []int, i int) {
	useSlice(a[i:])
	useSlice(a[i:]) // ERROR "Proved IsSliceInBounds$"
}

func f11c(a []int, i int) {
	useSlice(a[:i])
	useSlice(a[:i]) // ERROR "Proved IsSliceInBounds$"
}

func f11d(a []int, i int) {
	useInt(a[2*i+7])
	useInt(a[2*i+7]) // ERROR "Proved IsInBounds$"
}

func f12(a []int, b int) {
	useSlice(a[:b])
}

func f13a(a, b, c int, x bool) int {
	if a > 12 {
		if x {
			if a < 12 { // ERROR "Disproved Less64$"
				return 1
			}
		}
		if x {
			if a <= 12 { // ERROR "Disproved Leq64$"
				return 2
			}
		}
		if x {
			if a == 12 { // ERROR "Disproved Eq64$"
				return 3
			}
		}
		if x {
			if a >= 12 { // ERROR "Proved Leq64$"
				return 4
			}
		}
		if x {
			if a > 12 { // ERROR "Proved Less64$"
				return 5
			}
		}
		return 6
	}
	return 0
}

func f13b(a int, x bool) int {
	if a == -9 {
		if x {
			if a < -9 { // ERROR "Disproved Less64$"
				return 7
			}
		}
		if x {
			if a <= -9 { // ERROR "Proved Leq64$"
				return 8
			}
		}
		if x {
			if a == -9 { // ERROR "Proved Eq64$"
				return 9
			}
		}
		if x {
			if a >= -9 { // ERROR "Proved Leq64$"
				return 10
			}
		}
		if x {
			if a > -9 { // ERROR "Disproved Less64$"
				return 11
			}
		}
		return 12
	}
	return 0
}

func f13c(a int, x bool) int {
	if a < 90 {
		if x {
			if a < 90 { // ERROR "Proved Less64$"
				return 13
			}
		}
		if x {
			if a <= 90 { // ERROR "Proved Leq64$"
				return 14
			}
		}
		if x {
			if a == 90 { // ERROR "Disproved Eq64$"
				return 15
			}
		}
		if x {
			if a >= 90 { // ERROR "Disproved Leq64$"
				return 16
			}
		}
		if x {
			if a > 90 { // ERROR "Disproved Less64$"
				return 17
			}
		}
		return 18
	}
	return 0
}

func f13d(a int) int {
	if a < 5 {
		if a < 9 { // ERROR "Proved Less64$"
			return 1
		}
	}
	return 0
}

func f13e(a int) int {
	if a > 9 {
		if a > 5 { // ERROR "Proved Less64$"
			return 1
		}
	}
	return 0
}

func f13f(a, b int64) int64 {
	if b != math.MaxInt64 {
		return 42
	}
	if a > b { // ERROR "Disproved Less64$"
		if a == 0 {
			return 1
		}
	}
	return 0
}

func f13g(a int) int {
	if a < 3 {
		return 5
	}
	if a > 3 {
		return 6
	}
	if a == 3 { // ERROR "Proved Eq64$"
		return 7
	}
	return 8
}

func f13h(a int) int {
	if a < 3 {
		if a > 1 {
			if a == 2 { // ERROR "Proved Eq64$"
				return 5
			}
		}
	}
	return 0
}

func f13i(a uint) int {
	if a == 0 {
		return 1
	}
	if a > 0 { // ERROR "Proved Less64U$"
		return 2
	}
	return 3
}

func f14(p, q *int, a []int) {
	// This crazy ordering usually gives i1 the lowest value ID,
	// j the middle value ID, and i2 the highest value ID.
	// That used to confuse CSE because it ordered the args
	// of the two + ops below differently.
	// That in turn foiled bounds check elimination.
	i1 := *p
	j := *q
	i2 := *p
	useInt(a[i1+j])
	useInt(a[i2+j]) // ERROR "Proved IsInBounds$"
}

func f15(s []int, x int) {
	useSlice(s[x:])
	useSlice(s[:x]) // ERROR "Proved IsSliceInBounds$"
}

func f16(s []int) []int {
	if len(s) >= 10 {
		return s[:10] // ERROR "Proved IsSliceInBounds$"
	}
	return nil
}

func f17(b []int) {
	for i := 0; i < len(b); i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		// This tests for i <= cap, which we can only prove
		// using the derived relation between len and cap.
		// This depends on finding the contradiction, since we
		// don't query this condition directly.
		useSlice(b[:i]) // ERROR "Proved IsSliceInBounds$"
	}
}

func f18(b []int, x int, y uint) {
	_ = b[x]
	_ = b[y]

	if x > len(b) { // ERROR "Disproved Less64$"
		return
	}
	if y > uint(len(b)) { // ERROR "Disproved Less64U$"
		return
	}
	if int(y) > len(b) { // ERROR "Disproved Less64$"
		return
	}
}

func f19() (e int64, err error) {
	// Issue 29502: slice[:0] is incorrectly disproved.
	var stack []int64
	stack = append(stack, 123)
	if len(stack) > 1 {
		panic("too many elements")
	}
	last := len(stack) - 1
	e = stack[last]
	// Buggy compiler prints "Disproved Leq64" for the next line.
	stack = stack[:last]
	return e, nil
}

func sm1(b []int, x int) {
	// Test constant argument to slicemask.
	useSlice(b[2:8]) // optimized away earlier by rewrite
	// Test non-constant argument with known limits.
	if cap(b) > 10 {
		useSlice(b[2:]) // ERROR "Proved slicemask not needed$"
	}
}

func lim1(x, y, z int) {
	// Test relations between signed and unsigned limits.
	if x > 5 {
		if uint(x) > 5 { // ERROR "Proved Less64U$"
			return
		}
	}
	if y >= 0 && y < 4 {
		if uint(y) > 4 { // ERROR "Disproved Less64U$"
			return
		}
		if uint(y) < 5 { // ERROR "Proved Less64U$"
			return
		}
	}
	if z < 4 {
		if uint(z) > 4 { // Not provable without disjunctions.
			return
		}
	}
}

// fence1–4 correspond to the four fence-post implications.

func fence1(b []int, x, y int) {
	// Test proofs that rely on fence-post implications.
	if x+1 > y {
		if x < y { // ERROR "Disproved Less64$"
			return
		}
	}
	if len(b) < cap(b) {
		// This eliminates the growslice path.
		b = append(b, 1) // ERROR "Disproved Less64U$"
	}
}

func fence2(x, y int) {
	if x-1 < y {
		if x > y { // ERROR "Disproved Less64$"
			return
		}
	}
}

func fence3(b, c []int, x, y int64) {
	if x-1 >= y {
		if x <= y { // Can't prove because x may have wrapped.
			return
		}
	}

	if x != math.MinInt64 && x-1 >= y {
		if x <= y { // ERROR "Disproved Leq64$"
			return
		}
	}

	c[len(c)-1] = 0 // Can't prove because len(c) might be 0

	if n := len(b); n > 0 {
		b[n-1] = 0 // ERROR "Proved IsInBounds$"
	}
}

func fence4(x, y int64) {
	if x >= y+1 {
		if x <= y {
			return
		}
	}
	if y != math.MaxInt64 && x >= y+1 {
		if x <= y { // ERROR "Disproved Leq64$"
			return
		}
	}
}

// Check transitive relations
func trans1(x, y int64) {
	if x > 5 {
		if y > x {
			if y > 2 { // ERROR "Proved Less64$"
				return
			}
		} else if y == x {
			if y > 5 { // ERROR "Proved Less64$"
				return
			}
		}
	}
	if x >= 10 {
		if y > x {
			if y > 10 { // ERROR "Proved Less64$"
				return
			}
		}
	}
}

func trans2(a, b []int, i int) {
	if len(a) != len(b) {
		return
	}

	_ = a[i]
	_ = b[i] // ERROR "Proved IsInBounds$"
}

func trans3(a, b []int, i int) {
	if len(a) > len(b) {
		return
	}

	_ = a[i]
	_ = b[i] // ERROR "Proved IsInBounds$"
}

func trans4(b []byte, x int) {
	// Issue #42603: slice len/cap transitive relations.
	switch x {
	case 0:
		if len(b) < 20 {
			return
		}
		_ = b[:2] // ERROR "Proved IsSliceInBounds$"
	case 1:
		if len(b) < 40 {
			return
		}
		_ = b[:2] // ERROR "Proved IsSliceInBounds$"
	}
}

// Derived from nat.cmp
func natcmp(x, y []uint) (r int) {
	m := len(x)
	n := len(y)
	if m != n || m == 0 {
		return
	}

	i := m - 1
	for i > 0 && // ERROR "Induction variable: limits \(0,\?\], increment 1$"
		x[i] == // ERROR "Proved IsInBounds$"
			y[i] { // ERROR "Proved IsInBounds$"
		i--
	}

	switch {
	case x[i] < // todo, cannot prove this because it's dominated by i<=0 || x[i]==y[i]
		y[i]: // ERROR "Proved IsInBounds$"
		r = -1
	case x[i] > // ERROR "Proved IsInBounds$"
		y[i]: // ERROR "Proved IsInBounds$"
		r = 1
	}
	return
}

func suffix(s, suffix string) bool {
	// todo, we're still not able to drop the bound check here in the general case
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

func constsuffix(s string) bool {
	return suffix(s, "abc") // ERROR "Proved IsSliceInBounds$"
}

func atexit(foobar []func()) {
	for i := len(foobar) - 1; i >= 0; i-- { // ERROR "Induction variable: limits \[0,\?\], increment 1"
		f := foobar[i]
		foobar = foobar[:i] // ERROR "IsSliceInBounds"
		f()
	}
}

func make1(n int) []int {
	s := make([]int, n)
	for i := 0; i < n; i++ { // ERROR "Induction variable: limits \[0,\?\), increment 1"
		s[i] = 1 // ERROR "Proved IsInBounds$"
	}
	return s
}

func make2(n int) []int {
	s := make([]int, n)
	for i := range s { // ERROR "Induction variable: limits \[0,\?\), increment 1"
		s[i] = 1 // ERROR "Proved IsInBounds$"
	}
	return s
}

// The range tests below test the index variable of range loops.

// range1 compiles to the "efficiently indexable" form of a range loop.
func range1(b []int) {
	for i, v := range b { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		b[i] = v + 1    // ERROR "Proved IsInBounds$"
		if i < len(b) { // ERROR "Proved Less64$"
			println("x")
		}
		if i >= 0 { // ERROR "Proved Leq64$"
			println("x")
		}
	}
}

// range2 elements are larger, so they use the general form of a range loop.
func range2(b [][32]int) {
	for i, v := range b { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
		b[i][0] = v[0] + 1 // ERROR "Proved IsInBounds$"
		if i < len(b) {    // ERROR "Proved Less64$"
			println("x")
		}
		if i >= 0 { // ERROR "Proved Leq64$"
			println("x")
		}
	}
}

// signhint1-2 test whether the hint (int >= 0) is propagated into the loop.
func signHint1(i int, data []byte) {
	if i >= 0 {
		for i < len(data) { // ERROR "Induction variable: limits \[\?,\?\), increment 1$"
			_ = data[i] // ERROR "Proved IsInBounds$"
			i++
		}
	}
}

func signHint2(b []byte, n int) {
	if n < 0 {
		panic("")
	}
	_ = b[25]
	for i := n; i <= 25; i++ { // ERROR "Induction variable: limits \[\?,25\], increment 1$"
		b[i] = 123 // ERROR "Proved IsInBounds$"
	}
}

// indexGT0 tests whether prove learns int index >= 0 from bounds check.
func indexGT0(b []byte, n int) {
	_ = b[n]
	_ = b[25]

	for i := n; i <= 25; i++ { // ERROR "Induction variable: limits \[\?,25\], increment 1$"
		b[i] = 123 // ERROR "Proved IsInBounds$"
	}
}

// Induction variable in unrolled loop.
func unrollUpExcl(a []int) int {
	var i, x int
	for i = 0; i < len(a)-1; i += 2 { // ERROR "Induction variable: limits \[0,\?\), increment 2$"
		x += a[i]   // ERROR "Proved IsInBounds$"
		x += a[i+1] // ERROR "Proved IsInBounds( for blocked indexing)?$"
	}
	if i == len(a)-1 {
		x += a[i]
	}
	return x
}

// Induction variable in unrolled loop.
func unrollUpIncl(a []int) int {
	var i, x int
	for i = 0; i <= len(a)-2; i += 2 { // ERROR "Induction variable: limits \[0,\?\], increment 2$"
		x += a[i]   // ERROR "Proved IsInBounds$"
		x += a[i+1] // ERROR "Proved IsInBounds( for blocked indexing)?$"
	}
	if i == len(a)-1 {
		x += a[i]
	}
	return x
}

// Induction variable in unrolled loop.
func unrollDownExcl0(a []int) int {
	var i, x int
	for i = len(a) - 1; i > 0; i -= 2 { // ERROR "Induction variable: limits \(0,\?\], increment 2$"
		x += a[i]   // ERROR "Proved IsInBounds$"
		x += a[i-1] // ERROR "Proved IsInBounds$"
	}
	if i == 0 {
		x += a[i]
	}
	return x
}

// Induction variable in unrolled loop.
func unrollDownExcl1(a []int) int {
	var i, x int
	for i = len(a) - 1; i >= 1; i -= 2 { // ERROR "Induction variable: limits \(0,\?\], increment 2$"
		x += a[i]   // ERROR "Proved IsInBounds$"
		x += a[i-1] // ERROR "Proved IsInBounds$"
	}
	if i == 0 {
		x += a[i]
	}
	return x
}

// Induction variable in unrolled loop.
func unrollDownInclStep(a []int) int {
	var i, x int
	for i = len(a); i >= 2; i -= 2 { // ERROR "Induction variable: limits \[2,\?\], increment 2$"
		x += a[i-1] // ERROR "Proved IsInBounds$"
		x += a[i-2] // ERROR "Proved IsInBounds$"
	}
	if i == 1 {
		x += a[i-1]
	}
	return x
}

// Not an induction variable (step too large)
func unrollExclStepTooLarge(a []int) int {
	var i, x int
	for i = 0; i < len(a)-1; i += 3 {
		x += a[i]
		x += a[i+1] // ERROR "Proved IsInBounds( for blocked indexing)?$"
	}
	if i == len(a)-1 {
		x += a[i]
	}
	return x
}

// Not an induction variable (step too large)
func unrollInclStepTooLarge(a []int) int {
	var i, x int
	for i = 0; i <= len(a)-2; i += 3 {
		x += a[i]
		x += a[i+1] // ERROR "Proved IsInBounds( for blocked indexing)?$"
	}
	if i == len(a)-1 {
		x += a[i]
	}
	return x
}

// Not an induction variable (min too small, iterating down)
func unrollDecMin(a []int, b int) int {
	if b != math.MinInt64 {
		return 42
	}
	var i, x int
	for i = len(a); i >= b; i -= 2 { // ERROR "Proved Leq64"
		x += a[i-1]
		x += a[i-2]
	}
	if i == 1 {
		x += a[i-1]
	}
	return x
}

// Not an induction variable (min too small, iterating up -- perhaps could allow, but why bother?)
func unrollIncMin(a []int, b int) int {
	if b != math.MinInt64 {
		return 42
	}
	var i, x int
	for i = len(a); i >= b; i += 2 { // ERROR "Proved Leq64"
		x += a[i-1]
		x += a[i-2]
	}
	if i == 1 {
		x += a[i-1]
	}
	return x
}

// The 4 xxxxExtNto64 functions below test whether prove is looking
// through value-preserving sign/zero extensions of index values (issue #26292).

// Look through all extensions
func signExtNto64(x []int, j8 int8, j16 int16, j32 int32) int {
	if len(x) < 22 {
		return 0
	}
	if j8 >= 0 && j8 < 22 {
		return x[j8] // ERROR "Proved IsInBounds$"
	}
	if j16 >= 0 && j16 < 22 {
		return x[j16] // ERROR "Proved IsInBounds$"
	}
	if j32 >= 0 && j32 < 22 {
		return x[j32] // ERROR "Proved IsInBounds$"
	}
	return 0
}

func zeroExtNto64(x []int, j8 uint8, j16 uint16, j32 uint32) int {
	if len(x) < 22 {
		return 0
	}
	if j8 >= 0 && j8 < 22 {
		return x[j8] // ERROR "Proved IsInBounds$"
	}
	if j16 >= 0 && j16 < 22 {
		return x[j16] // ERROR "Proved IsInBounds$"
	}
	if j32 >= 0 && j32 < 22 {
		return x[j32] // ERROR "Proved IsInBounds$"
	}
	return 0
}

// Process fence-post implications through 32to64 extensions (issue #29964)
func signExt32to64Fence(x []int, j int32) int {
	if x[j] != 0 {
		return 1
	}
	if j > 0 && x[j-1] != 0 { // ERROR "Proved IsInBounds$"
		return 1
	}
	return 0
}

func zeroExt32to64Fence(x []int, j uint32) int {
	if x[j] != 0 {
		return 1
	}
	if j > 0 && x[j-1] != 0 { // ERROR "Proved IsInBounds$"
		return 1
	}
	return 0
}

// Ensure that bounds checks with negative indexes are not incorrectly removed.
func negIndex() {
	n := make([]int, 1)
	for i := -1; i <= 0; i++ { // ERROR "Induction variable: limits \[-1,0\], increment 1$"
		n[i] = 1
	}
}
func negIndex2(n int) {
	a := make([]int, 5)
	b := make([]int, 5)
	c := make([]int, 5)
	for i := -1; i <= 0; i-- {
		b[i] = i
		n++
		if n > 10 {
			break
		}
	}
	useSlice(a)
	useSlice(c)
}

// Check that prove is zeroing these right shifts of positive ints by bit-width - 1.
// e.g (Rsh64x64 <t> n (Const64 <typ.UInt64> [63])) && ft.isNonNegative(n) -> 0
func sh64(n int64) int64 {
	if n < 0 {
		return n
	}
	return n >> 63 // ERROR "Proved Rsh64x64 shifts to zero"
}

func sh32(n int32) int32 {
	if n < 0 {
		return n
	}
	return n >> 31 // ERROR "Proved Rsh32x64 shifts to zero"
}

func sh32x64(n int32) int32 {
	if n < 0 {
		return n
	}
	return n >> uint64(31) // ERROR "Proved Rsh32x64 shifts to zero"
}

func sh16(n int16) int16 {
	if n < 0 {
		return n
	}
	return n >> 15 // ERROR "Proved Rsh16x64 shifts to zero"
}

func sh64noopt(n int64) int64 {
	return n >> 63 // not optimized; n could be negative
}

// These cases are division of a positive signed integer by a power of 2.
// The opt pass doesnt have sufficient information to see that n is positive.
// So, instead, opt rewrites the division with a less-than-optimal replacement.
// Prove, which can see that n is nonnegative, cannot see the division because
// opt, an earlier pass, has already replaced it.
// The fix for this issue allows prove to zero a right shift that was added as
// part of the less-than-optimal reqwrite. That change by prove then allows
// lateopt to clean up all the unnecessary parts of the original division
// replacement. See issue #36159.
func divShiftClean(n int) int {
	if n < 0 {
		return n
	}
	return n / int(8) // ERROR "Proved Div64 is unsigned$"
}

func divShiftClean64(n int64) int64 {
	if n < 0 {
		return n
	}
	return n / int64(16)  // ERROR "Proved Div64 is unsigned$"
}

func divShiftClean32(n int32) int32 {
	if n < 0 {
		return n
	}
	return n / int32(16)  // ERROR "Proved Div32 is unsigned$"
}

// Bounds check elimination

func sliceBCE1(p []string, h uint) string {
	if len(p) == 0 {
		return ""
	}

	i := h & uint(len(p)-1)
	return p[i] // ERROR "Proved IsInBounds$"
}

func sliceBCE2(p []string, h int) string {
	if len(p) == 0 {
		return ""
	}
	i := h & (len(p) - 1)
	return p[i] // ERROR "Proved IsInBounds$"
}

func and(p []byte) ([]byte, []byte) { // issue #52563
	const blocksize = 16
	fullBlocks := len(p) &^ (blocksize - 1)
	blk := p[:fullBlocks] // ERROR "Proved IsSliceInBounds$"
	rem := p[fullBlocks:] // ERROR "Proved IsSliceInBounds$"
	return blk, rem
}

func rshu(x, y uint) int {
	z := x >> y
	if z <= x { // ERROR "Proved Leq64U$"
		return 1
	}
	return 0
}

func divu(x, y uint) int {
	z := x / y
	if z <= x { // ERROR "Proved Leq64U$"
		return 1
	}
	return 0
}

func divuRoundUp(x, y, z uint) int {
	x &= ^uint(0) >> 8 // can't overflow in add
	y = min(y, 0xff-1)
	z = max(z, 0xff)
	r := (x + y) / z // ERROR "Proved Neq64$"
	if r <= x {      // ERROR "Proved Leq64U$"
		return 1
	}
	return 0
}

func divuRoundUpSlice(x []string) {
	halfRoundedUp := uint(len(x)+1) / 2
	_ = x[:halfRoundedUp] // ERROR "Proved IsSliceInBounds$"
	_ = x[halfRoundedUp:] // ERROR "Proved IsSliceInBounds$"
}

func modu1(x, y uint) int {
	z := x % y
	if z < y { // ERROR "Proved Less64U$"
		return 1
	}
	return 0
}

func modu2(x, y uint) int {
	z := x % y
	if z <= x { // ERROR "Proved Leq64U$"
		return 1
	}
	return 0
}

func issue57077(s []int) (left, right []int) {
	middle := len(s) / 2 // ERROR "Proved Div64 is unsigned$"
	left = s[:middle]  // ERROR "Proved IsSliceInBounds$"
	right = s[middle:] // ERROR "Proved IsSliceInBounds$"
	return
}

func issue51622(b []byte) int {
	if len(b) >= 3 && b[len(b)-3] == '#' { // ERROR "Proved IsInBounds$"
		return len(b)
	}
	return 0
}

func issue45928(x int) {
	combinedFrac := x / (x | (1 << 31)) // ERROR "Proved Neq64$"
	useInt(combinedFrac)
}

func constantBounds1(i, j uint) int {
	var a [10]int
	if j < 11 && i < j {
		return a[i] // ERROR "Proved IsInBounds$"
	}
	return 0
}

func constantBounds2(i, j uint) int {
	var a [10]int
	if i < j && j < 11 {
		return a[i] // ERROR "Proved IsInBounds"
	}
	return 0
}

func constantBounds3(i, j, k, l uint) int {
	var a [8]int
	if i < j && j < k && k < l && l < 11 {
		return a[i] // ERROR "Proved IsInBounds"
	}
	return 0
}

func equalityPropagation(a [1]int, i, j uint) int {
	if i == j && i == 5 {
		return a[j-5] // ERROR "Proved IsInBounds"
	}
	return 0
}
func inequalityPropagation(a [1]int, i, j uint) int {
	if i != j && j >= 5 && j <= 6 && i == 5 {
		return a[j-6] // ERROR "Proved IsInBounds"
	}
	return 0
}

func issue66826a(a [21]byte) {
	for i := 0; i <= 10; i++ { // ERROR "Induction variable: limits \[0,10\], increment 1$"
		_ = a[2*i] // ERROR "Proved IsInBounds"
	}
}
func issue66826b(a [31]byte, i int) {
	if i < 0 || i > 10 {
		return
	}
	_ = a[3*i] // ERROR "Proved IsInBounds"
}

func f20(a, b bool) int {
	if a == b {
		if a {
			if b { // ERROR "Proved Arg"
				return 1
			}
		}
	}
	return 0
}

func f21(a, b *int) int {
	if a == b {
		if a != nil {
			if b != nil { // ERROR "Proved IsNonNil"
				return 1
			}
		}
	}
	return 0
}

func f22(b bool, x, y int) int {
	b2 := x < y
	if b == b2 {
		if b {
			if x >= y { // ERROR "Disproved Leq64$"
				return 1
			}
		}
	}
	return 0
}

func ctz64(x uint64, ensureBothBranchesCouldHappen bool) int {
	const max = math.MaxUint64
	sz := bits.Len64(max)

	log2half := uint64(max) >> (sz / 2)
	if x >= log2half || x == 0 {
		return 42
	}

	y := bits.TrailingZeros64(x) // ERROR "Proved Ctz64 non-zero$""

	z := sz / 2
	if ensureBothBranchesCouldHappen {
		if y < z { // ERROR "Proved Less64$"
			return -42
		}
	} else {
		if y >= z { // ERROR "Disproved Leq64$"
			return 1337
		}
	}

	return y
}
func ctz32(x uint32, ensureBothBranchesCouldHappen bool) int {
	const max = math.MaxUint32
	sz := bits.Len32(max)

	log2half := uint32(max) >> (sz / 2)
	if x >= log2half || x == 0 {
		return 42
	}

	y := bits.TrailingZeros32(x) // ERROR "Proved Ctz32 non-zero$""

	z := sz / 2
	if ensureBothBranchesCouldHappen {
		if y < z { // ERROR "Proved Less64$"
			return -42
		}
	} else {
		if y >= z { // ERROR "Disproved Leq64$"
			return 1337
		}
	}

	return y
}
func ctz16(x uint16, ensureBothBranchesCouldHappen bool) int {
	const max = math.MaxUint16
	sz := bits.Len16(max)

	log2half := uint16(max) >> (sz / 2)
	if x >= log2half || x == 0 {
		return 42
	}

	y := bits.TrailingZeros16(x) // ERROR "Proved Ctz16 non-zero$""

	z := sz / 2
	if ensureBothBranchesCouldHappen {
		if y < z { // ERROR "Proved Less64$"
			return -42
		}
	} else {
		if y >= z { // ERROR "Disproved Leq64$"
			return 1337
		}
	}

	return y
}
func ctz8(x uint8, ensureBothBranchesCouldHappen bool) int {
	const max = math.MaxUint8
	sz := bits.Len8(max)

	log2half := uint8(max) >> (sz / 2)
	if x >= log2half || x == 0 {
		return 42
	}

	y := bits.TrailingZeros8(x) // ERROR "Proved Ctz8 non-zero$""

	z := sz / 2
	if ensureBothBranchesCouldHappen {
		if y < z { // ERROR "Proved Less64$"
			return -42
		}
	} else {
		if y >= z { // ERROR "Disproved Leq64$"
			return 1337
		}
	}

	return y
}

func bitLen64(x uint64, ensureBothBranchesCouldHappen bool) int {
	const max = math.MaxUint64
	sz := bits.Len64(max)

	if x >= max>>3 {
		return 42
	}
	if x <= max>>6 {
		return 42
	}

	y := bits.Len64(x)

	if ensureBothBranchesCouldHappen {
		if sz-6 <= y && y <= sz-3 { // ERROR "Proved Leq64$"
			return -42
		}
	} else {
		if y < sz-6 || sz-3 < y { // ERROR "Disproved Less64$"
			return 1337
		}
	}
	return y
}
func bitLen32(x uint32, ensureBothBranchesCouldHappen bool) int {
	const max = math.MaxUint32
	sz := bits.Len32(max)

	if x >= max>>3 {
		return 42
	}
	if x <= max>>6 {
		return 42
	}

	y := bits.Len32(x)

	if ensureBothBranchesCouldHappen {
		if sz-6 <= y && y <= sz-3 { // ERROR "Proved Leq64$"
			return -42
		}
	} else {
		if y < sz-6 || sz-3 < y { // ERROR "Disproved Less64$"
			return 1337
		}
	}
	return y
}
func bitLen16(x uint16, ensureBothBranchesCouldHappen bool) int {
	const max = math.MaxUint16
	sz := bits.Len16(max)

	if x >= max>>3 {
		return 42
	}
	if x <= max>>6 {
		return 42
	}

	y := bits.Len16(x)

	if ensureBothBranchesCouldHappen {
		if sz-6 <= y && y <= sz-3 { // ERROR "Proved Leq64$"
			return -42
		}
	} else {
		if y < sz-6 || sz-3 < y { // ERROR "Disproved Less64$"
			return 1337
		}
	}
	return y
}
func bitLen8(x uint8, ensureBothBranchesCouldHappen bool) int {
	const max = math.MaxUint8
	sz := bits.Len8(max)

	if x >= max>>3 {
		return 42
	}
	if x <= max>>6 {
		return 42
	}

	y := bits.Len8(x)

	if ensureBothBranchesCouldHappen {
		if sz-6 <= y && y <= sz-3 { // ERROR "Proved Leq64$"
			return -42
		}
	} else {
		if y < sz-6 || sz-3 < y { // ERROR "Disproved Less64$"
			return 1337
		}
	}
	return y
}

func xor64(a, b uint64, ensureBothBranchesCouldHappen bool) int {
	a = min(a, 0xff)
	b = min(b, 0xfff)

	z := a ^ b

	if ensureBothBranchesCouldHappen {
		if z > 0xfff { // ERROR "Disproved Less64U$"
			return 42
		}
	} else {
		if z <= 0xfff { // ERROR "Proved Leq64U$"
			return 1337
		}
	}
	return int(z)
}

func or64(a, b uint64, ensureBothBranchesCouldHappen bool) int {
	a = min(a, 0xff)
	b = min(b, 0xfff)

	z := a | b

	if ensureBothBranchesCouldHappen {
		if z > 0xfff { // ERROR "Disproved Less64U$"
			return 42
		}
	} else {
		if z <= 0xfff { // ERROR "Proved Leq64U$"
			return 1337
		}
	}
	return int(z)
}

func mod64uWithSmallerDividendMax(a, b uint64, ensureBothBranchesCouldHappen bool) int {
	a = min(a, 0xff)
	b = min(b, 0xfff)

	z := bits.Len64(a % b) // see go.dev/issue/68857 for bits.Len64

	if ensureBothBranchesCouldHappen {
		if z > bits.Len64(0xff) { // ERROR "Disproved Less64$"
			return 42
		}
	} else {
		if z <= bits.Len64(0xff) { // ERROR "Proved Leq64$"
			return 1337
		}
	}
	return z
}
func mod64uWithSmallerDivisorMax(a, b uint64, ensureBothBranchesCouldHappen bool) int {
	a = min(a, 0xfff)
	b = min(b, 0x10) // we need bits.Len64(b.umax) != bits.Len64(b.umax-1)

	z := bits.Len64(a % b) // see go.dev/issue/68857 for bits.Len64

	if ensureBothBranchesCouldHappen {
		if z > bits.Len64(0x10-1) { // ERROR "Disproved Less64$"
			return 42
		}
	} else {
		if z <= bits.Len64(0x10-1) { // ERROR "Proved Leq64$"
			return 1337
		}
	}
	return z
}
func mod64uWithIdenticalMax(a, b uint64, ensureBothBranchesCouldHappen bool) int {
	a = min(a, 0x10)
	b = min(b, 0x10) // we need bits.Len64(b.umax) != bits.Len64(b.umax-1)

	z := bits.Len64(a % b) // see go.dev/issue/68857 for bits.Len64

	if ensureBothBranchesCouldHappen {
		if z > bits.Len64(0x10-1) { // ERROR "Disproved Less64$"
			return 42
		}
	} else {
		if z <= bits.Len64(0x10-1) { // ERROR "Proved Leq64$"
			return 1337
		}
	}
	return z
}
func mod64sPositiveWithSmallerDividendMax(a, b int64, ensureBothBranchesCouldHappen bool) int64 {
	if a < 0 || b < 0 {
		return 42
	}
	a = min(a, 0xff)
	b = min(b, 0xfff)

	z := a % b // ERROR "Proved Mod64 is unsigned$"

	if ensureBothBranchesCouldHappen {
		if z > 0xff { // ERROR "Disproved Less64$"
			return 42
		}
	} else {
		if z <= 0xff { // ERROR "Proved Leq64$"
			return 1337
		}
	}
	return z
}
func mod64sPositiveWithSmallerDivisorMax(a, b int64, ensureBothBranchesCouldHappen bool) int64 {
	if a < 0 || b < 0 {
		return 42
	}
	a = min(a, 0xfff)
	b = min(b, 0xff)

	z := a % b // ERROR "Proved Mod64 is unsigned$"

	if ensureBothBranchesCouldHappen {
		if z > 0xff-1 { // ERROR "Disproved Less64$"
			return 42
		}
	} else {
		if z <= 0xff-1 { // ERROR "Proved Leq64$"
			return 1337
		}
	}
	return z
}
func mod64sPositiveWithIdenticalMax(a, b int64, ensureBothBranchesCouldHappen bool) int64 {
	if a < 0 || b < 0 {
		return 42
	}
	a = min(a, 0xfff)
	b = min(b, 0xfff)

	z := a % b // ERROR "Proved Mod64 is unsigned$"

	if ensureBothBranchesCouldHappen {
		if z > 0xfff-1 { // ERROR "Disproved Less64$"
			return 42
		}
	} else {
		if z <= 0xfff-1 { // ERROR "Proved Leq64$"
			return 1337
		}
	}
	return z
}

func div64u(a, b uint64, ensureAllBranchesCouldHappen func() bool) uint64 {
	a = min(a, 0xffff)
	a = max(a, 0xfff)
	b = min(b, 0xff)
	b = max(b, 0xf)

	z := a / b // ERROR "Proved Neq64$"

	if ensureAllBranchesCouldHappen() && z > 0xffff/0xf { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z <= 0xffff/0xf { // ERROR "Proved Leq64U$"
		return 1337
	}
	if ensureAllBranchesCouldHappen() && z < 0xfff/0xff { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z >= 0xfff/0xff { // ERROR "Proved Leq64U$"
		return 42
	}
	return z
}
func div64s(a, b int64, ensureAllBranchesCouldHappen func() bool) int64 {
	if a < 0 || b < 0 {
		return 42
	}
	a = min(a, 0xffff)
	a = max(a, 0xfff)
	b = min(b, 0xff)
	b = max(b, 0xf)

	z := a / b // ERROR "Proved Div64 is unsigned|Proved Neq64"

	if ensureAllBranchesCouldHappen() && z > 0xffff/0xf { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z <= 0xffff/0xf { // ERROR "Proved Leq64$"
		return 1337
	}
	if ensureAllBranchesCouldHappen() && z < 0xfff/0xff { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z >= 0xfff/0xff { // ERROR "Proved Leq64$"
		return 42
	}
	return z
}

func trunc64to16(a uint64, ensureAllBranchesCouldHappen func() bool) uint16 {
	a = min(a, 0xfff)
	a = max(a, 0xff)

	z := uint16(a)
	if ensureAllBranchesCouldHappen() && z > 0xfff { // ERROR "Disproved Less16U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z <= 0xfff { // ERROR "Proved Leq16U$"
		return 1337
	}
	if ensureAllBranchesCouldHappen() && z < 0xff { // ERROR "Disproved Less16U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z >= 0xff { // ERROR "Proved Leq16U$"
		return 1337
	}
	return z
}

func com64(a uint64, ensureAllBranchesCouldHappen func() bool) uint64 {
	a = min(a, 0xffff)
	a = max(a, 0xff)

	z := ^a

	if ensureAllBranchesCouldHappen() && z > ^uint64(0xff) { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z <= ^uint64(0xff) { // ERROR "Proved Leq64U$"
		return 1337
	}
	if ensureAllBranchesCouldHappen() && z < ^uint64(0xffff) { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z >= ^uint64(0xffff) { // ERROR "Proved Leq64U$"
		return 1337
	}
	return z
}

func neg64(a uint64, ensureAllBranchesCouldHappen func() bool) uint64 {
	var lo, hi uint64 = 0xff, 0xfff
	a = min(a, hi)
	a = max(a, lo)

	z := -a

	if ensureAllBranchesCouldHappen() && z > -lo { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z <= -lo { // ERROR "Proved Leq64U$"
		return 1337
	}
	if ensureAllBranchesCouldHappen() && z < -hi { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z >= -hi { // ERROR "Proved Leq64U$"
		return 1337
	}
	return z
}
func neg64mightOverflowDuringNeg(a uint64, ensureAllBranchesCouldHappen func() bool) uint64 {
	var lo, hi uint64 = 0, 0xfff
	a = min(a, hi)
	a = max(a, lo)

	z := -a

	if ensureAllBranchesCouldHappen() && z > -lo {
		return 42
	}
	if ensureAllBranchesCouldHappen() && z <= -lo {
		return 1337
	}
	if ensureAllBranchesCouldHappen() && z < -hi {
		return 42
	}
	if ensureAllBranchesCouldHappen() && z >= -hi {
		return 1337
	}
	return z
}

func phiMin(a, b []byte) {
	_ = a[:min(len(a), len(b))] // ERROR "Proved IsSliceInBounds"
	_ = b[:min(len(a), len(b))] // ERROR "Proved IsSliceInBounds"
	_ = a[:max(len(a), len(b))]
	_ = b[:max(len(a), len(b))]
	x := len(a)
	if x > len(b) {
		x = len(b)
		useInt(0)
	}
	_ = a[:x] // ERROR "Proved IsSliceInBounds"
	y := len(a)
	if y > len(b) {
		y = len(b)
		useInt(0)
	} else {
		useInt(1)
	}
	_ = b[:y] // ERROR "Proved IsSliceInBounds"
}

func minPhiLeq[T uint | int](x, y T) (z T) {
	if x <= y {
		z = x
	} else {
		z = y
	}
	return z
}
func maxPhiLeq[T uint | int](x, y T) (z T) {
	if y <= x {
		z = x
	} else {
		z = y
	}
	return z
}
func mathBasedOnPhiLosangeMinUFirstLeq(x uint, ensureAllBranchesCouldHappen func() bool) uint {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = minPhiLeq(x, maxc)
	x = maxPhiLeq(x, minc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64U$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64U$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64U$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiLosangeMinUSecondLeq(x uint, ensureAllBranchesCouldHappen func() bool) uint {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = maxPhiLeq(x, minc)
	x = minPhiLeq(x, maxc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64U$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64U$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64U$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiLosangeMinFirstLeq(x int, ensureAllBranchesCouldHappen func() bool) int {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = minPhiLeq(x, maxc)
	x = maxPhiLeq(x, minc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiLosangeMinSecondLeq(x int, ensureAllBranchesCouldHappen func() bool) int {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = maxPhiLeq(x, minc)
	x = minPhiLeq(x, maxc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64$"
		return 42424242
	}
	return x
}

func minPhiLess[T uint | int](x, y T) (z T) {
	if x < y {
		z = x
	} else {
		z = y
	}
	return z
}
func maxPhiLess[T uint | int](x, y T) (z T) {
	if y < x {
		z = x
	} else {
		z = y
	}
	return z
}
func mathBasedOnPhiLosangeMinUFirstLess(x uint, ensureAllBranchesCouldHappen func() bool) uint {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = minPhiLess(x, maxc)
	x = maxPhiLess(x, minc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64U$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64U$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64U$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiLosangeMinUSecondLess(x uint, ensureAllBranchesCouldHappen func() bool) uint {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = maxPhiLess(x, minc)
	x = minPhiLess(x, maxc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64U$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64U$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64U$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiLosangeMinFirstLess(x int, ensureAllBranchesCouldHappen func() bool) int {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = minPhiLess(x, maxc)
	x = maxPhiLess(x, minc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiLosangeMinSecondLess(x int, ensureAllBranchesCouldHappen func() bool) int {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = maxPhiLess(x, minc)
	x = minPhiLess(x, maxc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64$"
		return 42424242
	}
	return x
}

func mathBasedOnPhiBuiltinMinUFirst(x uint, ensureAllBranchesCouldHappen func() bool) uint {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = min(x, maxc)
	x = max(x, minc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64U$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64U$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64U$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiBuiltinMinUSecond(x uint, ensureAllBranchesCouldHappen func() bool) uint {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = max(x, minc)
	x = min(x, maxc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64U$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64U$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64U$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64U$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiBuiltinMinFirst(x int, ensureAllBranchesCouldHappen func() bool) int {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = min(x, maxc)
	x = max(x, minc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64$"
		return 42424242
	}
	return x
}
func mathBasedOnPhiBuiltinMinSecond(x int, ensureAllBranchesCouldHappen func() bool) int {
	const maxc = 0xf2a
	const minc = 0xf0a
	x = max(x, minc)
	x = min(x, maxc)

	const k = 1
	x += k

	if ensureAllBranchesCouldHappen() && x > maxc+k { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && x <= maxc+k { // ERROR "Proved Leq64$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && x < minc+k { // ERROR "Disproved Less64$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && x >= minc+k { // ERROR "Proved Leq64$"
		return 42424242
	}
	return x
}

func issue16833(a, b []byte) {
	n := copy(a, b)
	_ = a[n:] // ERROR "Proved IsSliceInBounds"
	_ = b[n:] // ERROR "Proved IsSliceInBounds"
	_ = a[:n] // ERROR "Proved IsSliceInBounds"
	_ = b[:n] // ERROR "Proved IsSliceInBounds"
}

func clampedIdx1(x []int, i int) int {
	if len(x) == 0 {
		return 0
	}
	return x[min(max(0, i), len(x)-1)] // ERROR "Proved IsInBounds"
}
func clampedIdx2(x []int, i int) int {
	if len(x) == 0 {
		return 0
	}
	return x[max(min(i, len(x)-1), 0)] // TODO: can't get rid of this bounds check yet
}

func cvtBoolToUint8Disprove(b bool) byte {
	var c byte
	if b {
		c = 1
	}
	if c == 2 { // ERROR "Disproved Eq8"
		c = 3
	}
	return c
}
func cvtBoolToUint8BCE(b bool, a [2]int64) int64 {
	c := byte(0)
	if b {
		c = 1
	}
	return a[c] // ERROR "Proved IsInBounds$"
}

func transitiveProofsThroughNonOverflowingUnsignedAdd(x, y, z uint64) {
	x &= 1<<63 - 1
	y &= 1<<63 - 1

	a := x + y
	if a > z {
		return
	}

	if x > z { // ERROR "Disproved Less64U$"
		return
	}
	if y > z { // ERROR "Disproved Less64U$"
		return
	}
	if a == x {
		return
	}
	if a == y {
		return
	}

	x |= 1
	y |= 1
	a = x + y
	if a == x { // ERROR "Disproved Eq64$"
		return
	}
	if a == y { // ERROR "Disproved Eq64$"
		return
	}
}

func transitiveProofsThroughOverflowingUnsignedAdd(x, y, z uint64) {
	a := x + y
	if a > z {
		return
	}

	if x > z {
		return
	}
	if y > z {
		return
	}
	if a == x {
		return
	}
	if a == y {
		return
	}

	x |= 1
	y |= 1
	a = x + y
	if a == x {
		return
	}
	if a == y {
		return
	}
}

func transitiveProofsThroughNonOverflowingSignedAddPositive(x, y, z int64) {
	x &= 1<<62 - 1
	y &= 1<<62 - 1

	a := x + y
	if a > z {
		return
	}

	if x > z { // ERROR "Disproved Less64$"
		return
	}
	if y > z { // ERROR "Disproved Less64$"
		return
	}
	if a == x {
		return
	}
	if a == y {
		return
	}

	x |= 1
	y |= 1
	a = x + y
	if a == x { // ERROR "Disproved Eq64$"
		return
	}
	if a == y { // ERROR "Disproved Eq64$"
		return
	}
}

func transitiveProofsThroughOverflowingSignedAddPositive(x, y, z int64) {
	if x < 0 || y < 0 {
		return
	}

	a := x + y
	if a > z {
		return
	}

	if x > z {
		return
	}
	if y > z {
		return
	}
	if a == x {
		return
	}
	if a == y {
		return
	}

	x |= 1
	y |= 1
	a = x + y
	if a == x { // ERROR "Disproved Eq64$"
		return
	}
	if a == y { // ERROR "Disproved Eq64$"
		return
	}
}

func transitiveProofsThroughNonOverflowingSignedAddNegative(x, y, z int64) {
	if x < math.MinInt64>>1 || x > 0 ||
		y < math.MinInt64>>1 || y > 0 {
		return
	}

	a := x + y
	if a < z {
		return
	}

	if x < z { // ERROR "Disproved Less64$"
		return
	}
	if y < z { // ERROR "Disproved Less64$"
		return
	}
	if a == x {
		return
	}
	if a == y {
		return
	}

	if x == 0 && y == 0 {
		return
	}
	a = x + y
	if a == x { // ERROR "Disproved Eq64$"
		return
	}
	if a == y { // ERROR "Disproved Eq64$"
		return
	}
}

func transitiveProofsThroughOverflowingSignedAddNegative(x, y, z int64) {
	if x >= 0 || y >= 0 {
		return
	}

	a := x + y
	if a < z {
		return
	}

	if x < z {
		return
	}
	if y < z {
		return
	}
	if a == x {
		return
	}
	if a == y {
		return
	}

	x |= 1
	y |= 1
	a = x + y
	if a == x {
		return
	}
	if a == y {
		return
	}
}

func transitiveProofsThroughNonOverflowingUnsignedSub(x, y, z uint64) {
	x |= 0xfff
	y &= 0xfff

	a := x - y
	if a < z {
		return
	}

	if x < z { // ERROR "Disproved Less64U$"
		return
	}
	if y < z {
		return
	}
	if a == x {
		return
	}
	if a == y {
		return
	}

	y |= 1
	a = x - y
	if a == x { // ERROR "Disproved Eq64$"
		return
	}
	if a == y {
		return
	}
}

func transitiveProofsThroughOverflowingUnsignedSub(x, y, z uint64) {
	a := x - y
	if a < z {
		return
	}

	if x < z {
		return
	}
	if y < z {
		return
	}
	if a == x {
		return
	}
	if a == y {
		return
	}

	y |= 1
	a = x - y
	if a == x {
		return
	}
	if a == y {
		return
	}
}

func resliceString(s string) byte {
	if len(s) >= 4 {
		s = s[2:]   // ERROR "Proved IsSliceInBounds" "Proved slicemask not needed"
		s = s[1:]   // ERROR "Proved IsSliceInBounds" "Proved slicemask not needed"
		return s[0] // ERROR "Proved IsInBounds"
	}
	return 0
}
func resliceBytes(b []byte) byte {
	if len(b) >= 4 {
		b = b[2:]   // ERROR "Proved IsSliceInBounds" "Proved slicemask not needed"
		b = b[1:]   // ERROR "Proved IsSliceInBounds" "Proved slicemask not needed"
		return b[0] // ERROR "Proved IsInBounds"
	}
	return 0
}

func issue74473(s []uint) {
	i := 0
	for {
		if i >= len(s) { // ERROR "Induction variable: limits \[0,\?\), increment 1$"
			break
		}
		_ = s[i] // ERROR "Proved IsInBounds$"
		i++
	}
}

func setCapMaxBasedOnElementSize(x []uint64) int {
	c := uintptr(cap(x))
	max := ^uintptr(0) >> 3
	if c > max { // ERROR "Disproved Less"
		return 42
	}
	if c <= max { // ERROR "Proved Leq"
		return 1337
	}
	return 0
}

func issue75144for(a, b []uint64) bool {
	if len(a) == len(b) {
		for len(a) > 4 {
			a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			b = b[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
		}
		if len(a) == len(b) { // ERROR "Proved Eq64$"
			return true
		}
	}
	return false
}

func issue75144if(a, b []uint64) bool {
	if len(a) == len(b) {
		if len(a) > 4 {
			a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			b = b[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
		}
		if len(a) == len(b) { // ERROR "Proved Eq64$"
			return true
		}
	}
	return false
}

func issue75144if2(a, b, c, d []uint64) (r bool) {
	if len(a) != len(b) || len(c) != len(d) {
		return
	}
	if len(a) <= 4 || len(c) <= 4 {
		return
	}
	if len(a) < len(c) {
		c = c[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
		d = d[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
	} else {
		a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
		b = b[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
	}
	if len(a) == len(c) {
		return
	}
	if len(a) == len(b) { // ERROR "Proved Eq64$"
		r = true
	}
	if len(c) == len(d) { // ERROR "Proved Eq64$"
		r = true
	}
	return
}

func issue75144forCannot(a, b []uint64) bool {
	if len(a) == len(b) {
		for len(a) > 4 {
			a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			b = b[4:]
			for len(a) > 2 {
				a = a[2:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
				b = b[2:]
			}
		}
		if len(a) == len(b) {
			return true
		}
	}
	return false
}

func issue75144ifCannot(a, b []uint64) bool {
	if len(a) == len(b) {
		if len(a) > 4 {
			a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			b = b[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			if len(a) > 2 {
				a = a[2:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
				b = b[2:]
			}
		}
		if len(a) == len(b) {
			return true
		}
	}
	return false
}

func issue75144ifCannot2(a, b []uint64) bool {
	if len(a) == len(b) {
		if len(a) > 4 {
			a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			b = b[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
		} else if len(a) > 2 {
			a = a[2:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			b = b[2:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
		}
		if len(a) == len(b) {
			return true
		}
	}
	return false
}

func issue75144forNot(a, b []uint64) bool {
	if len(a) == len(b) {
		for len(a) > 4 {
			a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			b = b[3:]
		}
		if len(a) == len(b) {
			return true
		}
	}
	return false
}

func issue75144forNot2(a, b, c []uint64) bool {
	if len(a) == len(b) {
		for len(a) > 4 {
			a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
			b = c[4:]
		}
		if len(a) == len(b) {
			return true
		}
	}
	return false
}

func issue75144ifNot(a, b []uint64) bool {
	if len(a) == len(b) {
		if len(a) > 4 {
			a = a[4:] // ERROR "Proved slicemask not needed$" "Proved IsSliceInBounds$"
		} else {
			b = b[4:]
		}
		if len(a) == len(b) {
			return true
		}
	}
	return false
}

func mulIntoAnd(a, b uint) uint {
	if a > 1 || b > 1 {
		return 0
	}
	return a * b // ERROR "Rewrote Mul v[0-9]+ into And$"
}

func mulIntoCondSelect(a, b uint) uint {
	if a > 1 {
		return 0
	}
	return a * b // ERROR "Rewrote Mul v[0-9]+ into CondSelect"
}

func div7pos(x int32) bool {
	if x > 0 {
		return x%7 == 0 // ERROR "Proved Div32 is unsigned"
	}
	return false
}

func div2pos(x []int) int {
	return len(x) / 2 // ERROR "Proved Div64 is unsigned"
}

func div3pos(x []int) int {
	return len(x) / 3 // ERROR "Proved Div64 is unsigned"
}


var len200 [200]int

func modbound1(u uint64) int {
	s := 0
	for u > 0 {
		var d uint64
		u, d = u/100, u%100
		s += len200[d*2+1] // ERROR "Proved IsInBounds"
	}
	return s
}

func modbound2(p *[10]int, x uint) int {
	return p[x%9+1] // ERROR "Proved IsInBounds"
}

func shiftbound(x int) int {
	return 1 << (x % 11) // ERROR "Proved Lsh(32x32|64x64) bounded" "Proved Div64 does not need fix-up"
}

func shiftbound2(x int) int {
	return 1 << (x % 8) // ERROR "Proved Lsh(32x32|64x64) bounded" "Proved Div64 does not need fix-up"
}

func rangebound1(x []int) int {
	s := 0
	for i := range 1000 { // ERROR "Induction variable"
		if i < len(x) {
			s += x[i] // ERROR "Proved IsInBounds"
		}
	}
	return s
}

func rangebound2(x []int) int {
	s := 0
	if len(x) > 0 {
		for i := range 1000 { // ERROR "Induction variable"
			s += x[i%len(x)] // ERROR "Proved Mod64 is unsigned" "Proved Neq64" "Proved IsInBounds"
		}
	}
	return s
}

func swapbound(v []int) {
	for i := 0; i < len(v)/2; i++ { // ERROR "Proved Div64 is unsigned|Induction variable"
		v[i], // ERROR "Proved IsInBounds"
		v[len(v)-1-i] = // ERROR "Proved IsInBounds"
		v[len(v)-1-i],
		v[i] // ERROR "Proved IsInBounds"
	}
}

//go:noinline
func useInt(a int) {
}

//go:noinline
func useSlice(a []int) {
}

func main() {
}
