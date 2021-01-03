// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"fmt"
	"internal/testenv"
	"math/bits"
	"math/rand"
	"strings"
	"testing"
)

var isRaceBuilder = strings.HasSuffix(testenv.Builder(), "-race")

type funVV func(z, x, y []Word) (c Word)
type argVV struct {
	z, x, y nat
	c       Word
}

var sumVV = []argVV{
	{},
	{nat{0}, nat{0}, nat{0}, 0},
	{nat{1}, nat{1}, nat{0}, 0},
	{nat{0}, nat{_M}, nat{1}, 1},
	{nat{80235}, nat{12345}, nat{67890}, 0},
	{nat{_M - 1}, nat{_M}, nat{_M}, 1},
	{nat{0, 0, 0, 0}, nat{_M, _M, _M, _M}, nat{1, 0, 0, 0}, 1},
	{nat{0, 0, 0, _M}, nat{_M, _M, _M, _M - 1}, nat{1, 0, 0, 0}, 0},
	{nat{0, 0, 0, 0}, nat{_M, 0, _M, 0}, nat{1, _M, 0, _M}, 1},
}

func testFunVV(t *testing.T, msg string, f funVV, a argVV) {
	z := make(nat, len(a.z))
	c := f(z, a.x, a.y)
	for i, zi := range z {
		if zi != a.z[i] {
			t.Errorf("%s%+v\n\tgot z[%d] = %#x; want %#x", msg, a, i, zi, a.z[i])
			break
		}
	}
	if c != a.c {
		t.Errorf("%s%+v\n\tgot c = %#x; want %#x", msg, a, c, a.c)
	}
}

func TestFunVV(t *testing.T) {
	for _, a := range sumVV {
		arg := a
		testFunVV(t, "addVV_g", addVV_g, arg)
		testFunVV(t, "addVV", addVV, arg)

		arg = argVV{a.z, a.y, a.x, a.c}
		testFunVV(t, "addVV_g symmetric", addVV_g, arg)
		testFunVV(t, "addVV symmetric", addVV, arg)

		arg = argVV{a.x, a.z, a.y, a.c}
		testFunVV(t, "subVV_g", subVV_g, arg)
		testFunVV(t, "subVV", subVV, arg)

		arg = argVV{a.y, a.z, a.x, a.c}
		testFunVV(t, "subVV_g symmetric", subVV_g, arg)
		testFunVV(t, "subVV symmetric", subVV, arg)
	}
}

// Always the same seed for reproducible results.
var rnd = rand.New(rand.NewSource(0))

func rndW() Word {
	return Word(rnd.Int63()<<1 | rnd.Int63n(2))
}

func rndV(n int) []Word {
	v := make([]Word, n)
	for i := range v {
		v[i] = rndW()
	}
	return v
}

var benchSizes = []int{1, 2, 3, 4, 5, 1e1, 1e2, 1e3, 1e4, 1e5}

func BenchmarkAddVV(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		x := rndV(n)
		y := rndV(n)
		z := make([]Word, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _W))
			for i := 0; i < b.N; i++ {
				addVV(z, x, y)
			}
		})
	}
}

func BenchmarkSubVV(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		x := rndV(n)
		y := rndV(n)
		z := make([]Word, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _W))
			for i := 0; i < b.N; i++ {
				subVV(z, x, y)
			}
		})
	}
}

type funVW func(z, x []Word, y Word) (c Word)
type argVW struct {
	z, x nat
	y    Word
	c    Word
}

var sumVW = []argVW{
	{},
	{nil, nil, 2, 2},
	{nat{0}, nat{0}, 0, 0},
	{nat{1}, nat{0}, 1, 0},
	{nat{1}, nat{1}, 0, 0},
	{nat{0}, nat{_M}, 1, 1},
	{nat{0, 0, 0, 0}, nat{_M, _M, _M, _M}, 1, 1},
	{nat{585}, nat{314}, 271, 0},
}

var lshVW = []argVW{
	{},
	{nat{0}, nat{0}, 0, 0},
	{nat{0}, nat{0}, 1, 0},
	{nat{0}, nat{0}, 20, 0},

	{nat{_M}, nat{_M}, 0, 0},
	{nat{_M << 1 & _M}, nat{_M}, 1, 1},
	{nat{_M << 20 & _M}, nat{_M}, 20, _M >> (_W - 20)},

	{nat{_M, _M, _M}, nat{_M, _M, _M}, 0, 0},
	{nat{_M << 1 & _M, _M, _M}, nat{_M, _M, _M}, 1, 1},
	{nat{_M << 20 & _M, _M, _M}, nat{_M, _M, _M}, 20, _M >> (_W - 20)},
}

var rshVW = []argVW{
	{},
	{nat{0}, nat{0}, 0, 0},
	{nat{0}, nat{0}, 1, 0},
	{nat{0}, nat{0}, 20, 0},

	{nat{_M}, nat{_M}, 0, 0},
	{nat{_M >> 1}, nat{_M}, 1, _M << (_W - 1) & _M},
	{nat{_M >> 20}, nat{_M}, 20, _M << (_W - 20) & _M},

	{nat{_M, _M, _M}, nat{_M, _M, _M}, 0, 0},
	{nat{_M, _M, _M >> 1}, nat{_M, _M, _M}, 1, _M << (_W - 1) & _M},
	{nat{_M, _M, _M >> 20}, nat{_M, _M, _M}, 20, _M << (_W - 20) & _M},
}

func testFunVW(t *testing.T, msg string, f funVW, a argVW) {
	z := make(nat, len(a.z))
	c := f(z, a.x, a.y)
	for i, zi := range z {
		if zi != a.z[i] {
			t.Errorf("%s%+v\n\tgot z[%d] = %#x; want %#x", msg, a, i, zi, a.z[i])
			break
		}
	}
	if c != a.c {
		t.Errorf("%s%+v\n\tgot c = %#x; want %#x", msg, a, c, a.c)
	}
}

func testFunVWext(t *testing.T, msg string, f funVW, f_g funVW, a argVW) {
	// using the result of addVW_g/subVW_g as golden
	z_g := make(nat, len(a.z))
	c_g := f_g(z_g, a.x, a.y)
	c := f(a.z, a.x, a.y)

	for i, zi := range a.z {
		if zi != z_g[i] {
			t.Errorf("%s\n\tgot z[%d] = %#x; want %#x", msg, i, zi, z_g[i])
			break
		}
	}
	if c != c_g {
		t.Errorf("%s\n\tgot c = %#x; want %#x", msg, c, c_g)
	}
}

func makeFunVW(f func(z, x []Word, s uint) (c Word)) funVW {
	return func(z, x []Word, s Word) (c Word) {
		return f(z, x, uint(s))
	}
}

func TestFunVW(t *testing.T) {
	for _, a := range sumVW {
		arg := a
		testFunVW(t, "addVW_g", addVW_g, arg)
		testFunVW(t, "addVW", addVW, arg)

		arg = argVW{a.x, a.z, a.y, a.c}
		testFunVW(t, "subVW_g", subVW_g, arg)
		testFunVW(t, "subVW", subVW, arg)
	}

	shlVW_g := makeFunVW(shlVU_g)
	shlVW := makeFunVW(shlVU)
	for _, a := range lshVW {
		arg := a
		testFunVW(t, "shlVU_g", shlVW_g, arg)
		testFunVW(t, "shlVU", shlVW, arg)
	}

	shrVW_g := makeFunVW(shrVU_g)
	shrVW := makeFunVW(shrVU)
	for _, a := range rshVW {
		arg := a
		testFunVW(t, "shrVU_g", shrVW_g, arg)
		testFunVW(t, "shrVU", shrVW, arg)
	}
}

// Construct a vector comprising the same word, usually '0' or 'maximum uint'
func makeWordVec(e Word, n int) []Word {
	v := make([]Word, n)
	for i := range v {
		v[i] = e
	}
	return v
}

// Extended testing to addVW and subVW using various kinds of input data.
// We utilize the results of addVW_g and subVW_g as golden reference to check
// correctness.
func TestFunVWExt(t *testing.T) {
	// 32 is the current threshold that triggers an optimized version of
	// calculation for large-sized vector, ensure we have sizes around it tested.
	var vwSizes = []int{0, 1, 3, 4, 5, 8, 9, 23, 31, 32, 33, 34, 35, 36, 50, 120}
	for _, n := range vwSizes {
		// vector of random numbers, using the result of addVW_g/subVW_g as golden
		x := rndV(n)
		y := rndW()
		z := make(nat, n)
		arg := argVW{z, x, y, 0}
		testFunVWext(t, "addVW, random inputs", addVW, addVW_g, arg)
		testFunVWext(t, "subVW, random inputs", subVW, subVW_g, arg)

		// vector of random numbers, but make 'x' and 'z' share storage
		arg = argVW{x, x, y, 0}
		testFunVWext(t, "addVW, random inputs, sharing storage", addVW, addVW_g, arg)
		testFunVWext(t, "subVW, random inputs, sharing storage", subVW, subVW_g, arg)

		// vector of maximum uint, to force carry flag set in each 'add'
		y = ^Word(0)
		x = makeWordVec(y, n)
		arg = argVW{z, x, y, 0}
		testFunVWext(t, "addVW, vector of max uint", addVW, addVW_g, arg)

		// vector of '0', to force carry flag set in each 'sub'
		x = makeWordVec(0, n)
		arg = argVW{z, x, 1, 0}
		testFunVWext(t, "subVW, vector of zero", subVW, subVW_g, arg)
	}
}

type argVU struct {
	d  []Word // d is a Word slice, the input parameters x and z come from this array.
	l  uint   // l is the length of the input parameters x and z.
	xp uint   // xp is the starting position of the input parameter x, x := d[xp:xp+l].
	zp uint   // zp is the starting position of the input parameter z, z := d[zp:zp+l].
	s  uint   // s is the shift number.
	r  []Word // r is the expected output result z.
	c  Word   // c is the expected return value.
	m  string // message.
}

var argshlVUIn = []Word{1, 2, 4, 8, 16, 32, 64, 0, 0, 0}
var argshlVUr0 = []Word{1, 2, 4, 8, 16, 32, 64}
var argshlVUr1 = []Word{2, 4, 8, 16, 32, 64, 128}
var argshlVUrWm1 = []Word{1 << (_W - 1), 0, 1, 2, 4, 8, 16}

var argshlVU = []argVU{
	// test cases for shlVU
	{[]Word{1, _M, _M, _M, _M, _M, 3 << (_W - 2), 0}, 7, 0, 0, 1, []Word{2, _M - 1, _M, _M, _M, _M, 1<<(_W-1) + 1}, 1, "complete overlap of shlVU"},
	{[]Word{1, _M, _M, _M, _M, _M, 3 << (_W - 2), 0, 0, 0, 0}, 7, 0, 3, 1, []Word{2, _M - 1, _M, _M, _M, _M, 1<<(_W-1) + 1}, 1, "partial overlap by half of shlVU"},
	{[]Word{1, _M, _M, _M, _M, _M, 3 << (_W - 2), 0, 0, 0, 0, 0, 0, 0}, 7, 0, 6, 1, []Word{2, _M - 1, _M, _M, _M, _M, 1<<(_W-1) + 1}, 1, "partial overlap by 1 Word of shlVU"},
	{[]Word{1, _M, _M, _M, _M, _M, 3 << (_W - 2), 0, 0, 0, 0, 0, 0, 0, 0}, 7, 0, 7, 1, []Word{2, _M - 1, _M, _M, _M, _M, 1<<(_W-1) + 1}, 1, "no overlap of shlVU"},
	// additional test cases with shift values of 0, 1 and (_W-1)
	{argshlVUIn, 7, 0, 0, 0, argshlVUr0, 0, "complete overlap of shlVU and shift of 0"},
	{argshlVUIn, 7, 0, 0, 1, argshlVUr1, 0, "complete overlap of shlVU and shift of 1"},
	{argshlVUIn, 7, 0, 0, _W - 1, argshlVUrWm1, 32, "complete overlap of shlVU and shift of _W - 1"},
	{argshlVUIn, 7, 0, 1, 0, argshlVUr0, 0, "partial overlap by 6 Words of shlVU and shift of 0"},
	{argshlVUIn, 7, 0, 1, 1, argshlVUr1, 0, "partial overlap by 6 Words of shlVU and shift of 1"},
	{argshlVUIn, 7, 0, 1, _W - 1, argshlVUrWm1, 32, "partial overlap by 6 Words of shlVU and shift of _W - 1"},
	{argshlVUIn, 7, 0, 2, 0, argshlVUr0, 0, "partial overlap by 5 Words of shlVU and shift of 0"},
	{argshlVUIn, 7, 0, 2, 1, argshlVUr1, 0, "partial overlap by 5 Words of shlVU and shift of 1"},
	{argshlVUIn, 7, 0, 2, _W - 1, argshlVUrWm1, 32, "partial overlap by 5 Words of shlVU abd shift of _W - 1"},
	{argshlVUIn, 7, 0, 3, 0, argshlVUr0, 0, "partial overlap by 4 Words of shlVU and shift of 0"},
	{argshlVUIn, 7, 0, 3, 1, argshlVUr1, 0, "partial overlap by 4 Words of shlVU and shift of 1"},
	{argshlVUIn, 7, 0, 3, _W - 1, argshlVUrWm1, 32, "partial overlap by 4 Words of shlVU and shift of _W - 1"},
}

var argshrVUIn = []Word{0, 0, 0, 1, 2, 4, 8, 16, 32, 64}
var argshrVUr0 = []Word{1, 2, 4, 8, 16, 32, 64}
var argshrVUr1 = []Word{0, 1, 2, 4, 8, 16, 32}
var argshrVUrWm1 = []Word{4, 8, 16, 32, 64, 128, 0}

var argshrVU = []argVU{
	// test cases for shrVU
	{[]Word{0, 3, _M, _M, _M, _M, _M, 1 << (_W - 1)}, 7, 1, 1, 1, []Word{1<<(_W-1) + 1, _M, _M, _M, _M, _M >> 1, 1 << (_W - 2)}, 1 << (_W - 1), "complete overlap of shrVU"},
	{[]Word{0, 0, 0, 0, 3, _M, _M, _M, _M, _M, 1 << (_W - 1)}, 7, 4, 1, 1, []Word{1<<(_W-1) + 1, _M, _M, _M, _M, _M >> 1, 1 << (_W - 2)}, 1 << (_W - 1), "partial overlap by half of shrVU"},
	{[]Word{0, 0, 0, 0, 0, 0, 0, 3, _M, _M, _M, _M, _M, 1 << (_W - 1)}, 7, 7, 1, 1, []Word{1<<(_W-1) + 1, _M, _M, _M, _M, _M >> 1, 1 << (_W - 2)}, 1 << (_W - 1), "partial overlap by 1 Word of shrVU"},
	{[]Word{0, 0, 0, 0, 0, 0, 0, 0, 3, _M, _M, _M, _M, _M, 1 << (_W - 1)}, 7, 8, 1, 1, []Word{1<<(_W-1) + 1, _M, _M, _M, _M, _M >> 1, 1 << (_W - 2)}, 1 << (_W - 1), "no overlap of shrVU"},
	// additional test cases with shift values of 0, 1 and (_W-1)
	{argshrVUIn, 7, 3, 3, 0, argshrVUr0, 0, "complete overlap of shrVU and shift of 0"},
	{argshrVUIn, 7, 3, 3, 1, argshrVUr1, 1 << (_W - 1), "complete overlap of shrVU and shift of 1"},
	{argshrVUIn, 7, 3, 3, _W - 1, argshrVUrWm1, 2, "complete overlap of shrVU and shift of _W - 1"},
	{argshrVUIn, 7, 3, 2, 0, argshrVUr0, 0, "partial overlap by 6 Words of shrVU and shift of 0"},
	{argshrVUIn, 7, 3, 2, 1, argshrVUr1, 1 << (_W - 1), "partial overlap by 6 Words of shrVU and shift of 1"},
	{argshrVUIn, 7, 3, 2, _W - 1, argshrVUrWm1, 2, "partial overlap by 6 Words of shrVU and shift of _W - 1"},
	{argshrVUIn, 7, 3, 1, 0, argshrVUr0, 0, "partial overlap by 5 Words of shrVU and shift of 0"},
	{argshrVUIn, 7, 3, 1, 1, argshrVUr1, 1 << (_W - 1), "partial overlap by 5 Words of shrVU and shift of 1"},
	{argshrVUIn, 7, 3, 1, _W - 1, argshrVUrWm1, 2, "partial overlap by 5 Words of shrVU and shift of _W - 1"},
	{argshrVUIn, 7, 3, 0, 0, argshrVUr0, 0, "partial overlap by 4 Words of shrVU and shift of 0"},
	{argshrVUIn, 7, 3, 0, 1, argshrVUr1, 1 << (_W - 1), "partial overlap by 4 Words of shrVU and shift of 1"},
	{argshrVUIn, 7, 3, 0, _W - 1, argshrVUrWm1, 2, "partial overlap by 4 Words of shrVU and shift of _W - 1"},
}

func testShiftFunc(t *testing.T, f func(z, x []Word, s uint) Word, a argVU) {
	// work on copy of a.d to preserve the original data.
	b := make([]Word, len(a.d))
	copy(b, a.d)
	z := b[a.zp : a.zp+a.l]
	x := b[a.xp : a.xp+a.l]
	c := f(z, x, a.s)
	for i, zi := range z {
		if zi != a.r[i] {
			t.Errorf("d := %v, %s(d[%d:%d], d[%d:%d], %d)\n\tgot z[%d] = %#x; want %#x", a.d, a.m, a.zp, a.zp+a.l, a.xp, a.xp+a.l, a.s, i, zi, a.r[i])
			break
		}
	}
	if c != a.c {
		t.Errorf("d := %v, %s(d[%d:%d], d[%d:%d], %d)\n\tgot c = %#x; want %#x", a.d, a.m, a.zp, a.zp+a.l, a.xp, a.xp+a.l, a.s, c, a.c)
	}
}

func TestShiftOverlap(t *testing.T) {
	for _, a := range argshlVU {
		arg := a
		testShiftFunc(t, shlVU, arg)
	}

	for _, a := range argshrVU {
		arg := a
		testShiftFunc(t, shrVU, arg)
	}
}

func TestIssue31084(t *testing.T) {
	// compute 10^n via 5^n << n.
	const n = 165
	p := nat(nil).expNN(nat{5}, nat{n}, nil)
	p = p.shl(p, n)
	got := string(p.utoa(10))
	want := "1" + strings.Repeat("0", n)
	if got != want {
		t.Errorf("shl(%v, %v)\n\tgot  %s\n\twant %s", p, n, got, want)
	}
}

const issue42838Value = "159309191113245227702888039776771180559110455519261878607388585338616290151305816094308987472018268594098344692611135542392730712890625"

func TestIssue42838(t *testing.T) {
	const s = 192
	z, _, _, _ := nat(nil).scan(strings.NewReader(issue42838Value), 0, false)
	z = z.shl(z, s)
	got := string(z.utoa(10))
	want := "1" + strings.Repeat("0", s)
	if got != want {
		t.Errorf("shl(%v, %v)\n\tgot  %s\n\twant %s", z, s, got, want)
	}
}

func BenchmarkAddVW(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		x := rndV(n)
		y := rndW()
		z := make([]Word, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _S))
			for i := 0; i < b.N; i++ {
				addVW(z, x, y)
			}
		})
	}
}

// Benchmarking addVW using vector of maximum uint to force carry flag set
func BenchmarkAddVWext(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		y := ^Word(0)
		x := makeWordVec(y, n)
		z := make([]Word, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _S))
			for i := 0; i < b.N; i++ {
				addVW(z, x, y)
			}
		})
	}
}

func BenchmarkSubVW(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		x := rndV(n)
		y := rndW()
		z := make([]Word, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _S))
			for i := 0; i < b.N; i++ {
				subVW(z, x, y)
			}
		})
	}
}

// Benchmarking subVW using vector of zero to force carry flag set
func BenchmarkSubVWext(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		x := makeWordVec(0, n)
		y := Word(1)
		z := make([]Word, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _S))
			for i := 0; i < b.N; i++ {
				subVW(z, x, y)
			}
		})
	}
}

type funVWW func(z, x []Word, y, r Word) (c Word)
type argVWW struct {
	z, x nat
	y, r Word
	c    Word
}

var prodVWW = []argVWW{
	{},
	{nat{0}, nat{0}, 0, 0, 0},
	{nat{991}, nat{0}, 0, 991, 0},
	{nat{0}, nat{_M}, 0, 0, 0},
	{nat{991}, nat{_M}, 0, 991, 0},
	{nat{0}, nat{0}, _M, 0, 0},
	{nat{991}, nat{0}, _M, 991, 0},
	{nat{1}, nat{1}, 1, 0, 0},
	{nat{992}, nat{1}, 1, 991, 0},
	{nat{22793}, nat{991}, 23, 0, 0},
	{nat{22800}, nat{991}, 23, 7, 0},
	{nat{0, 0, 0, 22793}, nat{0, 0, 0, 991}, 23, 0, 0},
	{nat{7, 0, 0, 22793}, nat{0, 0, 0, 991}, 23, 7, 0},
	{nat{0, 0, 0, 0}, nat{7893475, 7395495, 798547395, 68943}, 0, 0, 0},
	{nat{991, 0, 0, 0}, nat{7893475, 7395495, 798547395, 68943}, 0, 991, 0},
	{nat{0, 0, 0, 0}, nat{0, 0, 0, 0}, 894375984, 0, 0},
	{nat{991, 0, 0, 0}, nat{0, 0, 0, 0}, 894375984, 991, 0},
	{nat{_M << 1 & _M}, nat{_M}, 1 << 1, 0, _M >> (_W - 1)},
	{nat{_M<<1&_M + 1}, nat{_M}, 1 << 1, 1, _M >> (_W - 1)},
	{nat{_M << 7 & _M}, nat{_M}, 1 << 7, 0, _M >> (_W - 7)},
	{nat{_M<<7&_M + 1<<6}, nat{_M}, 1 << 7, 1 << 6, _M >> (_W - 7)},
	{nat{_M << 7 & _M, _M, _M, _M}, nat{_M, _M, _M, _M}, 1 << 7, 0, _M >> (_W - 7)},
	{nat{_M<<7&_M + 1<<6, _M, _M, _M}, nat{_M, _M, _M, _M}, 1 << 7, 1 << 6, _M >> (_W - 7)},
}

func testFunVWW(t *testing.T, msg string, f funVWW, a argVWW) {
	z := make(nat, len(a.z))
	c := f(z, a.x, a.y, a.r)
	for i, zi := range z {
		if zi != a.z[i] {
			t.Errorf("%s%+v\n\tgot z[%d] = %#x; want %#x", msg, a, i, zi, a.z[i])
			break
		}
	}
	if c != a.c {
		t.Errorf("%s%+v\n\tgot c = %#x; want %#x", msg, a, c, a.c)
	}
}

// TODO(gri) mulAddVWW and divWVW are symmetric operations but
//           their signature is not symmetric. Try to unify.

type funWVW func(z []Word, xn Word, x []Word, y Word) (r Word)
type argWVW struct {
	z  nat
	xn Word
	x  nat
	y  Word
	r  Word
}

func testFunWVW(t *testing.T, msg string, f funWVW, a argWVW) {
	z := make(nat, len(a.z))
	r := f(z, a.xn, a.x, a.y)
	for i, zi := range z {
		if zi != a.z[i] {
			t.Errorf("%s%+v\n\tgot z[%d] = %#x; want %#x", msg, a, i, zi, a.z[i])
			break
		}
	}
	if r != a.r {
		t.Errorf("%s%+v\n\tgot r = %#x; want %#x", msg, a, r, a.r)
	}
}

func TestFunVWW(t *testing.T) {
	for _, a := range prodVWW {
		arg := a
		testFunVWW(t, "mulAddVWW_g", mulAddVWW_g, arg)
		testFunVWW(t, "mulAddVWW", mulAddVWW, arg)

		if a.y != 0 && a.r < a.y {
			arg := argWVW{a.x, a.c, a.z, a.y, a.r}
			testFunWVW(t, "divWVW", divWVW, arg)
		}
	}
}

var mulWWTests = []struct {
	x, y Word
	q, r Word
}{
	{_M, _M, _M - 1, 1},
	// 32 bit only: {0xc47dfa8c, 50911, 0x98a4, 0x998587f4},
}

func TestMulWW(t *testing.T) {
	for i, test := range mulWWTests {
		q, r := mulWW_g(test.x, test.y)
		if q != test.q || r != test.r {
			t.Errorf("#%d got (%x, %x) want (%x, %x)", i, q, r, test.q, test.r)
		}
	}
}

var mulAddWWWTests = []struct {
	x, y, c Word
	q, r    Word
}{
	// TODO(agl): These will only work on 64-bit platforms.
	// {15064310297182388543, 0xe7df04d2d35d5d80, 13537600649892366549, 13644450054494335067, 10832252001440893781},
	// {15064310297182388543, 0xdab2f18048baa68d, 13644450054494335067, 12869334219691522700, 14233854684711418382},
	{_M, _M, 0, _M - 1, 1},
	{_M, _M, _M, _M, 0},
}

func TestMulAddWWW(t *testing.T) {
	for i, test := range mulAddWWWTests {
		q, r := mulAddWWW_g(test.x, test.y, test.c)
		if q != test.q || r != test.r {
			t.Errorf("#%d got (%x, %x) want (%x, %x)", i, q, r, test.q, test.r)
		}
	}
}

var divWWTests = []struct {
	x1, x0, y Word
	q, r      Word
}{
	{_M >> 1, 0, _M, _M >> 1, _M >> 1},
	{_M - (1 << (_W - 2)), _M, 3 << (_W - 2), _M, _M - (1 << (_W - 2))},
}

const testsNumber = 1 << 16

func TestDivWW(t *testing.T) {
	i := 0
	for i, test := range divWWTests {
		rec := reciprocalWord(test.y)
		q, r := divWW(test.x1, test.x0, test.y, rec)
		if q != test.q || r != test.r {
			t.Errorf("#%d got (%x, %x) want (%x, %x)", i, q, r, test.q, test.r)
		}
	}
	//random tests
	for ; i < testsNumber; i++ {
		x1 := rndW()
		x0 := rndW()
		y := rndW()
		if x1 >= y {
			continue
		}
		rec := reciprocalWord(y)
		qGot, rGot := divWW(x1, x0, y, rec)
		qWant, rWant := bits.Div(uint(x1), uint(x0), uint(y))
		if uint(qGot) != qWant || uint(rGot) != rWant {
			t.Errorf("#%d got (%x, %x) want (%x, %x)", i, qGot, rGot, qWant, rWant)
		}
	}
}

func BenchmarkMulAddVWW(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		z := make([]Word, n+1)
		x := rndV(n)
		y := rndW()
		r := rndW()
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _W))
			for i := 0; i < b.N; i++ {
				mulAddVWW(z, x, y, r)
			}
		})
	}
}

func BenchmarkAddMulVVW(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		x := rndV(n)
		y := rndW()
		z := make([]Word, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _W))
			for i := 0; i < b.N; i++ {
				addMulVVW(z, x, y)
			}
		})
	}
}
func BenchmarkDivWVW(b *testing.B) {
	for _, n := range benchSizes {
		if isRaceBuilder && n > 1e3 {
			continue
		}
		x := rndV(n)
		y := rndW()
		z := make([]Word, n)
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			b.SetBytes(int64(n * _W))
			for i := 0; i < b.N; i++ {
				divWVW(z, 0, x, y)
			}
		})
	}
}
