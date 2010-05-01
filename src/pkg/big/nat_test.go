// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "testing"

type cmpTest struct {
	x, y nat
	r    int
}


var cmpTests = []cmpTest{
	cmpTest{nil, nil, 0},
	cmpTest{nil, nat{}, 0},
	cmpTest{nat{}, nil, 0},
	cmpTest{nat{}, nat{}, 0},
	cmpTest{nat{0}, nat{0}, 0},
	cmpTest{nat{0}, nat{1}, -1},
	cmpTest{nat{1}, nat{0}, 1},
	cmpTest{nat{1}, nat{1}, 0},
	cmpTest{nat{0, _M}, nat{1}, 1},
	cmpTest{nat{1}, nat{0, _M}, -1},
	cmpTest{nat{1, _M}, nat{0, _M}, 1},
	cmpTest{nat{0, _M}, nat{1, _M}, -1},
	cmpTest{nat{16, 571956, 8794, 68}, nat{837, 9146, 1, 754489}, -1},
	cmpTest{nat{34986, 41, 105, 1957}, nat{56, 7458, 104, 1957}, 1},
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
	argNN{},
	argNN{nat{1}, nil, nat{1}},
	argNN{nat{1111111110}, nat{123456789}, nat{987654321}},
	argNN{nat{0, 0, 0, 1}, nil, nat{0, 0, 0, 1}},
	argNN{nat{0, 0, 0, 1111111110}, nat{0, 0, 0, 123456789}, nat{0, 0, 0, 987654321}},
	argNN{nat{0, 0, 0, 1}, nat{0, 0, _M}, nat{0, 0, 1}},
}


var prodNN = []argNN{
	argNN{},
	argNN{nil, nil, nil},
	argNN{nil, nat{991}, nil},
	argNN{nat{991}, nat{991}, nat{1}},
	argNN{nat{991 * 991}, nat{991}, nat{991}},
	argNN{nat{0, 0, 991 * 991}, nat{0, 991}, nat{0, 991}},
	argNN{nat{1 * 991, 2 * 991, 3 * 991, 4 * 991}, nat{1, 2, 3, 4}, nat{991}},
	argNN{nat{4, 11, 20, 30, 20, 11, 4}, nat{1, 2, 3, 4}, nat{4, 3, 2, 1}},
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


type mulRange struct {
	a, b uint64
	prod string
}


var mulRanges = []mulRange{
	mulRange{0, 0, "0"},
	mulRange{1, 1, "1"},
	mulRange{1, 2, "2"},
	mulRange{1, 3, "6"},
	mulRange{1, 3, "6"},
	mulRange{10, 10, "10"},
	mulRange{0, 100, "0"},
	mulRange{0, 1e9, "0"},
	mulRange{100, 1, "1"},                  // empty range
	mulRange{1, 10, "3628800"},             // 10!
	mulRange{1, 20, "2432902008176640000"}, // 20!
	mulRange{1, 100,
		"933262154439441526816992388562667004907159682643816214685929" +
			"638952175999932299156089414639761565182862536979208272237582" +
			"51185210916864000000000000000000000000", // 100!
	},
}


func TestMulRange(t *testing.T) {
	for i, r := range mulRanges {
		prod := nat(nil).mulRange(r.a, r.b).string(10)
		if prod != r.prod {
			t.Errorf("%d: got %s; want %s", i, prod, r.prod)
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


type strN struct {
	x nat
	b int
	s string
}


var tabN = []strN{
	strN{nil, 10, "0"},
	strN{nat{1}, 10, "1"},
	strN{nat{10}, 10, "10"},
	strN{nat{1234567890}, 10, "1234567890"},
}


func TestString(t *testing.T) {
	for _, a := range tabN {
		s := a.x.string(a.b)
		if s != a.s {
			t.Errorf("stringN%+v\n\tgot s = %s; want %s", a, s, a.s)
		}

		x, b, n := nat(nil).scan(a.s, a.b)
		if x.cmp(a.x) != 0 {
			t.Errorf("scanN%+v\n\tgot z = %v; want %v", a, x, a.x)
		}
		if b != a.b {
			t.Errorf("scanN%+v\n\tgot b = %d; want %d", a, b, a.b)
		}
		if n != len(a.s) {
			t.Errorf("scanN%+v\n\tgot n = %d; want %d", a, n, len(a.s))
		}
	}
}


func TestLeadingZeroBits(t *testing.T) {
	var x Word = 1 << (_W - 1)
	for i := 0; i <= _W; i++ {
		if leadingZeroBits(x) != i {
			t.Errorf("failed at %x: got %d want %d", x, leadingZeroBits(x), i)
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
	shiftTest{nil, 0, nil},
	shiftTest{nil, 1, nil},
	shiftTest{natOne, 0, natOne},
	shiftTest{natOne, 1, natTwo},
	shiftTest{nat{1 << (_W - 1)}, 1, nat{0}},
	shiftTest{nat{1 << (_W - 1), 0}, 1, nat{0, 1}},
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
	shiftTest{nil, 0, nil},
	shiftTest{nil, 1, nil},
	shiftTest{natOne, 0, natOne},
	shiftTest{natOne, 1, nil},
	shiftTest{natTwo, 1, natOne},
	shiftTest{nat{0, 1}, 1, nat{1 << (_W - 1)}},
	shiftTest{nat{2, 1, 1}, 1, nat{1<<(_W-1) + 1, 1 << (_W - 1)}},
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
	modWTest{"23492635982634928349238759823742", "252341", "220170"},
}


var modWTests64 = []modWTest{
	modWTest{"6527895462947293856291561095690465243862946", "524326975699234", "375066989628668"},
}


func runModWTests(t *testing.T, tests []modWTest) {
	for i, test := range tests {
		in, _ := new(Int).SetString(test.in, 10)
		d, _ := new(Int).SetString(test.dividend, 10)
		out, _ := new(Int).SetString(test.out, 10)

		r := in.abs.modW(d.abs[0])
		if r != out.abs[0] {
			t.Errorf("#%d failed: got %s want %s\n", i, r, out)
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
			t.Errorf("Failed at step %d: x: %x got: %d\n", i, x, trailingZeroBits(x))
		}
		x <<= 1
	}
}


type expNNTest struct {
	x, y, m string
	out     string
}


var expNNTests = []expNNTest{
	expNNTest{"0x8000000000000000", "2", "", "0x40000000000000000000000000000000"},
	expNNTest{"0x8000000000000000", "2", "6719", "4944"},
	expNNTest{"0x8000000000000000", "3", "6719", "5447"},
	expNNTest{"0x8000000000000000", "1000", "6719", "1603"},
	expNNTest{"0x8000000000000000", "1000000", "6719", "3199"},
	expNNTest{
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
