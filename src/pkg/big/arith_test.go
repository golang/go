// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "testing"

type funWW func(x, y, c Word) (z1, z0 Word)
type argWW struct {
	x, y, c, z1, z0 Word
}

var sumWW = []argWW{
	{0, 0, 0, 0, 0},
	{0, 1, 0, 0, 1},
	{0, 0, 1, 0, 1},
	{0, 1, 1, 0, 2},
	{12345, 67890, 0, 0, 80235},
	{12345, 67890, 1, 0, 80236},
	{_M, 1, 0, 1, 0},
	{_M, 0, 1, 1, 0},
	{_M, 1, 1, 1, 1},
	{_M, _M, 0, 1, _M - 1},
	{_M, _M, 1, 1, _M},
}

func testFunWW(t *testing.T, msg string, f funWW, a argWW) {
	z1, z0 := f(a.x, a.y, a.c)
	if z1 != a.z1 || z0 != a.z0 {
		t.Errorf("%s%+v\n\tgot z1:z0 = %#x:%#x; want %#x:%#x", msg, a, z1, z0, a.z1, a.z0)
	}
}

func TestFunWW(t *testing.T) {
	for _, a := range sumWW {
		arg := a
		testFunWW(t, "addWW_g", addWW_g, arg)

		arg = argWW{a.y, a.x, a.c, a.z1, a.z0}
		testFunWW(t, "addWW_g symmetric", addWW_g, arg)

		arg = argWW{a.z0, a.x, a.c, a.z1, a.y}
		testFunWW(t, "subWW_g", subWW_g, arg)

		arg = argWW{a.z0, a.y, a.c, a.z1, a.x}
		testFunWW(t, "subWW_g symmetric", subWW_g, arg)
	}
}

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
}

var prodVW = []argVW{
	{},
	{nat{0}, nat{0}, 0, 0},
	{nat{0}, nat{_M}, 0, 0},
	{nat{0}, nat{0}, _M, 0},
	{nat{1}, nat{1}, 1, 0},
	{nat{22793}, nat{991}, 23, 0},
	{nat{0, 0, 0, 22793}, nat{0, 0, 0, 991}, 23, 0},
	{nat{0, 0, 0, 0}, nat{7893475, 7395495, 798547395, 68943}, 0, 0},
	{nat{0, 0, 0, 0}, nat{0, 0, 0, 0}, 894375984, 0},
	{nat{_M << 1 & _M}, nat{_M}, 1 << 1, _M >> (_W - 1)},
	{nat{_M << 7 & _M}, nat{_M}, 1 << 7, _M >> (_W - 7)},
	{nat{_M << 7 & _M, _M, _M, _M}, nat{_M, _M, _M, _M}, 1 << 7, _M >> (_W - 7)},
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
			testFunWVW(t, "divWVW_g", divWVW_g, arg)
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
