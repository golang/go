// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "testing"

var cmpTests = []struct {
	x, y nat
	r    int
}{
	{nil, nil, 0},
	{nil, nat{}, 0},
	{nat{}, nil, 0},
	{nat{}, nat{}, 0},
	{nat{0}, nat{0}, 0},
	{nat{0}, nat{1}, -1},
	{nat{1}, nat{0}, 1},
	{nat{1}, nat{1}, 0},
	{nat{0, _M}, nat{1}, 1},
	{nat{1}, nat{0, _M}, -1},
	{nat{1, _M}, nat{0, _M}, 1},
	{nat{0, _M}, nat{1, _M}, -1},
	{nat{16, 571956, 8794, 68}, nat{837, 9146, 1, 754489}, -1},
	{nat{34986, 41, 105, 1957}, nat{56, 7458, 104, 1957}, 1},
}


func TestCmp(t *testing.T) {
	for i, a := range cmpTests {
		r := a.x.cmp(a.y)
		if r != a.r {
			t.Errorf("#%d got r = %v; want %v", i, r, a.r)
		}
	}
}


type funNN func(z, x, y nat) nat
type argNN struct {
	z, x, y nat
}


var sumNN = []argNN{
	{},
	{nat{1}, nil, nat{1}},
	{nat{1111111110}, nat{123456789}, nat{987654321}},
	{nat{0, 0, 0, 1}, nil, nat{0, 0, 0, 1}},
	{nat{0, 0, 0, 1111111110}, nat{0, 0, 0, 123456789}, nat{0, 0, 0, 987654321}},
	{nat{0, 0, 0, 1}, nat{0, 0, _M}, nat{0, 0, 1}},
}


var prodNN = []argNN{
	{},
	{nil, nil, nil},
	{nil, nat{991}, nil},
	{nat{991}, nat{991}, nat{1}},
	{nat{991 * 991}, nat{991}, nat{991}},
	{nat{0, 0, 991 * 991}, nat{0, 991}, nat{0, 991}},
	{nat{1 * 991, 2 * 991, 3 * 991, 4 * 991}, nat{1, 2, 3, 4}, nat{991}},
	{nat{4, 11, 20, 30, 20, 11, 4}, nat{1, 2, 3, 4}, nat{4, 3, 2, 1}},
}


func TestSet(t *testing.T) {
	for _, a := range sumNN {
		z := nat(nil).set(a.z)
		if z.cmp(a.z) != 0 {
			t.Errorf("got z = %v; want %v", z, a.z)
		}
	}
}


func testFunNN(t *testing.T, msg string, f funNN, a argNN) {
	z := f(nil, a.x, a.y)
	if z.cmp(a.z) != 0 {
		t.Errorf("%s%+v\n\tgot z = %v; want %v", msg, a, z, a.z)
	}
}


func TestFunNN(t *testing.T) {
	for _, a := range sumNN {
		arg := a
		testFunNN(t, "add", nat.add, arg)

		arg = argNN{a.z, a.y, a.x}
		testFunNN(t, "add symmetric", nat.add, arg)

		arg = argNN{a.x, a.z, a.y}
		testFunNN(t, "sub", nat.sub, arg)

		arg = argNN{a.y, a.z, a.x}
		testFunNN(t, "sub symmetric", nat.sub, arg)
	}

	for _, a := range prodNN {
		arg := a
		testFunNN(t, "mul", nat.mul, arg)

		arg = argNN{a.z, a.y, a.x}
		testFunNN(t, "mul symmetric", nat.mul, arg)
	}
}


var mulRangesN = []struct {
	a, b uint64
	prod string
}{
	{0, 0, "0"},
	{1, 1, "1"},
	{1, 2, "2"},
	{1, 3, "6"},
	{10, 10, "10"},
	{0, 100, "0"},
	{0, 1e9, "0"},
	{1, 0, "1"},                    // empty range
	{100, 1, "1"},                  // empty range
	{1, 10, "3628800"},             // 10!
	{1, 20, "2432902008176640000"}, // 20!
	{1, 100,
		"933262154439441526816992388562667004907159682643816214685929" +
			"638952175999932299156089414639761565182862536979208272237582" +
			"51185210916864000000000000000000000000", // 100!
	},
}


func TestMulRangeN(t *testing.T) {
	for i, r := range mulRangesN {
		prod := nat(nil).mulRange(r.a, r.b).string(10)
		if prod != r.prod {
			t.Errorf("#%d: got %s; want %s", i, prod, r.prod)
		}
	}
}


var mulArg, mulTmp nat

func init() {
	const n = 1000
	mulArg = make(nat, n)
	for i := 0; i < n; i++ {
		mulArg[i] = _M
	}
}


func benchmarkMulLoad() {
	for j := 1; j <= 10; j++ {
		x := mulArg[0 : j*100]
		mulTmp.mul(x, x)
	}
}


func BenchmarkMul(b *testing.B) {
	for i := 0; i < b.N; i++ {
		benchmarkMulLoad()
	}
}


var tab = []struct {
	x nat
	b int
	s string
}{
	{nil, 10, "0"},
	{nat{1}, 10, "1"},
	{nat{10}, 10, "10"},
	{nat{1234567890}, 10, "1234567890"},
}


func TestString(t *testing.T) {
	for _, a := range tab {
		s := a.x.string(a.b)
		if s != a.s {
			t.Errorf("string%+v\n\tgot s = %s; want %s", a, s, a.s)
		}

		x, b, n := nat(nil).scan(a.s, a.b)
		if x.cmp(a.x) != 0 {
			t.Errorf("scan%+v\n\tgot z = %v; want %v", a, x, a.x)
		}
		if b != a.b {
			t.Errorf("scan%+v\n\tgot b = %d; want %d", a, b, a.b)
		}
		if n != len(a.s) {
			t.Errorf("scan%+v\n\tgot n = %d; want %d", a, n, len(a.s))
		}
	}
}


func TestLeadingZeros(t *testing.T) {
	var x Word = _B >> 1
	for i := 0; i <= _W; i++ {
		if int(leadingZeros(x)) != i {
			t.Errorf("failed at %x: got %d want %d", x, leadingZeros(x), i)
		}
		x >>= 1
	}
}


type shiftTest struct {
	in    nat
	shift uint
	out   nat
}


var leftShiftTests = []shiftTest{
	{nil, 0, nil},
	{nil, 1, nil},
	{natOne, 0, natOne},
	{natOne, 1, natTwo},
	{nat{1 << (_W - 1)}, 1, nat{0}},
	{nat{1 << (_W - 1), 0}, 1, nat{0, 1}},
}


func TestShiftLeft(t *testing.T) {
	for i, test := range leftShiftTests {
		var z nat
		z = z.shl(test.in, test.shift)
		for j, d := range test.out {
			if j >= len(z) || z[j] != d {
				t.Errorf("#%d: got: %v want: %v", i, z, test.out)
				break
			}
		}
	}
}


var rightShiftTests = []shiftTest{
	{nil, 0, nil},
	{nil, 1, nil},
	{natOne, 0, natOne},
	{natOne, 1, nil},
	{natTwo, 1, natOne},
	{nat{0, 1}, 1, nat{1 << (_W - 1)}},
	{nat{2, 1, 1}, 1, nat{1<<(_W-1) + 1, 1 << (_W - 1)}},
}


func TestShiftRight(t *testing.T) {
	for i, test := range rightShiftTests {
		var z nat
		z = z.shr(test.in, test.shift)
		for j, d := range test.out {
			if j >= len(z) || z[j] != d {
				t.Errorf("#%d: got: %v want: %v", i, z, test.out)
				break
			}
		}
	}
}


type modWTest struct {
	in       string
	dividend string
	out      string
}


var modWTests32 = []modWTest{
	{"23492635982634928349238759823742", "252341", "220170"},
}


var modWTests64 = []modWTest{
	{"6527895462947293856291561095690465243862946", "524326975699234", "375066989628668"},
}


func runModWTests(t *testing.T, tests []modWTest) {
	for i, test := range tests {
		in, _ := new(Int).SetString(test.in, 10)
		d, _ := new(Int).SetString(test.dividend, 10)
		out, _ := new(Int).SetString(test.out, 10)

		r := in.abs.modW(d.abs[0])
		if r != out.abs[0] {
			t.Errorf("#%d failed: got %s want %s", i, r, out)
		}
	}
}


func TestModW(t *testing.T) {
	if _W >= 32 {
		runModWTests(t, modWTests32)
	}
	if _W >= 64 {
		runModWTests(t, modWTests64)
	}
}


func TestTrailingZeroBits(t *testing.T) {
	var x Word
	x--
	for i := 0; i < _W; i++ {
		if trailingZeroBits(x) != i {
			t.Errorf("Failed at step %d: x: %x got: %d", i, x, trailingZeroBits(x))
		}
		x <<= 1
	}
}


var expNNTests = []struct {
	x, y, m string
	out     string
}{
	{"0x8000000000000000", "2", "", "0x40000000000000000000000000000000"},
	{"0x8000000000000000", "2", "6719", "4944"},
	{"0x8000000000000000", "3", "6719", "5447"},
	{"0x8000000000000000", "1000", "6719", "1603"},
	{"0x8000000000000000", "1000000", "6719", "3199"},
	{
		"2938462938472983472983659726349017249287491026512746239764525612965293865296239471239874193284792387498274256129746192347",
		"298472983472983471903246121093472394872319615612417471234712061",
		"29834729834729834729347290846729561262544958723956495615629569234729836259263598127342374289365912465901365498236492183464",
		"23537740700184054162508175125554701713153216681790245129157191391322321508055833908509185839069455749219131480588829346291",
	},
}


func TestExpNN(t *testing.T) {
	for i, test := range expNNTests {
		x, _, _ := nat(nil).scan(test.x, 0)
		y, _, _ := nat(nil).scan(test.y, 0)
		out, _, _ := nat(nil).scan(test.out, 0)

		var m nat

		if len(test.m) > 0 {
			m, _, _ = nat(nil).scan(test.m, 0)
		}

		z := nat(nil).expNN(x, y, m)
		if z.cmp(out) != 0 {
			t.Errorf("#%d got %v want %v", i, z, out)
		}
	}
}
