// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes";
	"encoding/hex";
	"testing";
	"testing/quick";
)

func newZ(x int64) *Int {
	var z Int;
	return z.New(x);
}


type funZZ func(z, x, y *Int) *Int
type argZZ struct {
	z, x, y *Int;
}


var sumZZ = []argZZ{
	argZZ{newZ(0), newZ(0), newZ(0)},
	argZZ{newZ(1), newZ(1), newZ(0)},
	argZZ{newZ(1111111110), newZ(123456789), newZ(987654321)},
	argZZ{newZ(-1), newZ(-1), newZ(0)},
	argZZ{newZ(864197532), newZ(-123456789), newZ(987654321)},
	argZZ{newZ(-1111111110), newZ(-123456789), newZ(-987654321)},
}


var prodZZ = []argZZ{
	argZZ{newZ(0), newZ(0), newZ(0)},
	argZZ{newZ(0), newZ(1), newZ(0)},
	argZZ{newZ(1), newZ(1), newZ(1)},
	argZZ{newZ(-991 * 991), newZ(991), newZ(-991)},
	// TODO(gri) add larger products
}


func TestSetZ(t *testing.T) {
	for _, a := range sumZZ {
		var z Int;
		z.Set(a.z);
		if CmpInt(&z, a.z) != 0 {
			t.Errorf("got z = %v; want %v", z, a.z);
		}
	}
}


func testFunZZ(t *testing.T, msg string, f funZZ, a argZZ) {
	var z Int;
	f(&z, a.x, a.y);
	if CmpInt(&z, a.z) != 0 {
		t.Errorf("%s%+v\n\tgot z = %v; want %v", msg, a, &z, a.z);
	}
}


func TestSumZZ(t *testing.T) {
	AddZZ := func(z, x, y *Int) *Int { return z.Add(x, y) };
	SubZZ := func(z, x, y *Int) *Int { return z.Sub(x, y) };
	for _, a := range sumZZ {
		arg := a;
		testFunZZ(t, "AddZZ", AddZZ, arg);

		arg = argZZ{a.z, a.y, a.x};
		testFunZZ(t, "AddZZ symmetric", AddZZ, arg);

		arg = argZZ{a.x, a.z, a.y};
		testFunZZ(t, "SubZZ", SubZZ, arg);

		arg = argZZ{a.y, a.z, a.x};
		testFunZZ(t, "SubZZ symmetric", SubZZ, arg);
	}
}


func TestProdZZ(t *testing.T) {
	MulZZ := func(z, x, y *Int) *Int { return z.Mul(x, y) };
	for _, a := range prodZZ {
		arg := a;
		testFunZZ(t, "MulZZ", MulZZ, arg);

		arg = argZZ{a.z, a.y, a.x};
		testFunZZ(t, "MulZZ symmetric", MulZZ, arg);
	}
}


var facts = map[int]string{
	0: "1",
	1: "1",
	2: "2",
	10: "3628800",
	20: "2432902008176640000",
	100: "933262154439441526816992388562667004907159682643816214685929"
	"638952175999932299156089414639761565182862536979208272237582"
	"51185210916864000000000000000000000000",
}


func fact(n int) *Int {
	var z Int;
	z.New(1);
	for i := 2; i <= n; i++ {
		var t Int;
		t.New(int64(i));
		z.Mul(&z, &t);
	}
	return &z;
}


func TestFact(t *testing.T) {
	for n, s := range facts {
		f := fact(n).String();
		if f != s {
			t.Errorf("%d! = %s; want %s", n, f, s);
		}
	}
}


type fromStringTest struct {
	in	string;
	base	int;
	out	int64;
	ok	bool;
}


var fromStringTests = []fromStringTest{
	fromStringTest{in: "", ok: false},
	fromStringTest{in: "a", ok: false},
	fromStringTest{in: "z", ok: false},
	fromStringTest{in: "+", ok: false},
	fromStringTest{"0", 0, 0, true},
	fromStringTest{"0", 10, 0, true},
	fromStringTest{"0", 16, 0, true},
	fromStringTest{"10", 0, 10, true},
	fromStringTest{"10", 10, 10, true},
	fromStringTest{"10", 16, 16, true},
	fromStringTest{"-10", 16, -16, true},
	fromStringTest{in: "0x", ok: false},
	fromStringTest{"0x10", 0, 16, true},
	fromStringTest{in: "0x10", base: 16, ok: false},
	fromStringTest{"-0x10", 0, -16, true},
}


func TestSetString(t *testing.T) {
	for i, test := range fromStringTests {
		n, ok := new(Int).SetString(test.in, test.base);
		if ok != test.ok {
			t.Errorf("#%d (input '%s') ok incorrect (should be %t)", i, test.in, test.ok);
			continue;
		}
		if !ok {
			continue;
		}

		if CmpInt(n, new(Int).New(test.out)) != 0 {
			t.Errorf("#%d (input '%s') got: %s want: %d\n", i, test.in, n, test.out);
		}
	}
}


type divSignsTest struct {
	x, y	int64;
	q, r	int64;
}


// These examples taken from the Go Language Spec, section "Arithmetic operators"
var divSignsTests = []divSignsTest{
	divSignsTest{5, 3, 1, 2},
	divSignsTest{-5, 3, -1, -2},
	divSignsTest{5, -3, -1, 2},
	divSignsTest{-5, -3, 1, -2},
	divSignsTest{1, 2, 0, 1},
}


func TestDivSigns(t *testing.T) {
	for i, test := range divSignsTests {
		x := new(Int).New(test.x);
		y := new(Int).New(test.y);
		q, r := new(Int).Div(x, y);
		expectedQ := new(Int).New(test.q);
		expectedR := new(Int).New(test.r);

		if CmpInt(q, expectedQ) != 0 || CmpInt(r, expectedR) != 0 {
			t.Errorf("#%d: got (%s, %s) want (%s, %s)", i, q, r, expectedQ, expectedR);
		}
	}
}


func checkSetBytes(b []byte) bool {
	hex1 := hex.EncodeToString(new(Int).SetBytes(b).Bytes());
	hex2 := hex.EncodeToString(b);

	for len(hex1) < len(hex2) {
		hex1 = "0"+hex1;
	}

	for len(hex1) > len(hex2) {
		hex2 = "0"+hex2;
	}

	return hex1 == hex2;
}


func TestSetBytes(t *testing.T) {
	err := quick.Check(checkSetBytes, nil);
	if err != nil {
		t.Error(err);
	}
}


func checkBytes(b []byte) bool {
	b2 := new(Int).SetBytes(b).Bytes();
	return bytes.Compare(b, b2) == 0;
}


func TestBytes(t *testing.T) {
	err := quick.Check(checkSetBytes, nil);
	if err != nil {
		t.Error(err);
	}
}


func checkDiv(x, y []byte) bool {
	u := new(Int).SetBytes(x);
	v := new(Int).SetBytes(y);

	if len(v.abs) == 0 {
		return true;
	}

	q, r := new(Int).Div(u, v);

	if CmpInt(r, v) >= 0 {
		return false;
	}

	uprime := new(Int).Set(q);
	uprime.Mul(uprime, v);
	uprime.Add(uprime, r);

	return CmpInt(uprime, u) == 0;
}


func TestDiv(t *testing.T) {
	err := quick.Check(checkDiv, nil);
	if err != nil {
		t.Error(err);
	}
}


func TestDivStepD6(t *testing.T) {
	// See Knuth, Volume 2, section 4.3.1, exercise 21. This code exercises
	// a code path which only triggers 1 in 10^{-19} cases.

	u := &Int{false, []Word{0, 0, 0x8000000000000001, 0x7fffffffffffffff}};
	v := &Int{false, []Word{5, 0x8000000000000002, 0x8000000000000000}};

	q, r := new(Int).Div(u, v);
	const expectedQ = "18446744073709551613";
	const expectedR = "3138550867693340382088035895064302439801311770021610913807";
	if q.String() != expectedQ || r.String() != expectedR {
		t.Errorf("got (%s, %s) want (%s, %s)", q, r, expectedQ, expectedR);
	}
}


type lenTest struct {
	in	string;
	out	int;
}


var lenTests = []lenTest{
	lenTest{"0", 0},
	lenTest{"1", 1},
	lenTest{"2", 2},
	lenTest{"4", 3},
	lenTest{"0x8000", 16},
	lenTest{"0x80000000", 32},
	lenTest{"0x800000000000", 48},
	lenTest{"0x8000000000000000", 64},
	lenTest{"0x80000000000000000000", 80},
}


func TestLen(t *testing.T) {
	for i, test := range lenTests {
		n, ok := new(Int).SetString(test.in, 0);
		if !ok {
			t.Errorf("#%d test input invalid: %s", i, test.in);
			continue;
		}

		if n.Len() != test.out {
			t.Errorf("#%d got %d want %d\n", i, n.Len(), test.out);
		}
	}
}


type expTest struct {
	x, y, m	string;
	out	string;
}


var expTests = []expTest{
	/*expTest{"5", "0", "", "1"},
	expTest{"-5", "0", "", "-1"},
	expTest{"5", "1", "", "5"},
	expTest{"-5", "1", "", "-5"},
	expTest{"5", "2", "", "25"},*/
	expTest{"1", "65537", "2", "1"},
	/*expTest{"0x8000000000000000", "2", "", "0x40000000000000000000000000000000"},
	expTest{"0x8000000000000000", "2", "6719", "4944"},
	expTest{"0x8000000000000000", "3", "6719", "5447"},
	expTest{"0x8000000000000000", "1000", "6719", "1603"},
	expTest{"0x8000000000000000", "1000000", "6719", "3199"},
	expTest{
		"2938462938472983472983659726349017249287491026512746239764525612965293865296239471239874193284792387498274256129746192347",
		"298472983472983471903246121093472394872319615612417471234712061",
		"29834729834729834729347290846729561262544958723956495615629569234729836259263598127342374289365912465901365498236492183464",
		"23537740700184054162508175125554701713153216681790245129157191391322321508055833908509185839069455749219131480588829346291",
	},*/
}


func TestExp(t *testing.T) {
	for i, test := range expTests {
		x, ok1 := new(Int).SetString(test.x, 0);
		y, ok2 := new(Int).SetString(test.y, 0);
		out, ok3 := new(Int).SetString(test.out, 0);

		var ok4 bool;
		var m *Int;

		if len(test.m) == 0 {
			m, ok4 = nil, true;
		} else {
			m, ok4 = new(Int).SetString(test.m, 0);
		}

		if !ok1 || !ok2 || !ok3 || !ok4 {
			t.Errorf("#%d error in input", i);
			continue;
		}

		z := new(Int).Exp(x, y, m);
		if CmpInt(z, out) != 0 {
			t.Errorf("#%d got %s want %s", i, z, out);
		}
	}
}


func checkGcd(aBytes, bBytes []byte) bool {
	a := new(Int).SetBytes(aBytes);
	b := new(Int).SetBytes(bBytes);

	x := new(Int);
	y := new(Int);
	d := new(Int);

	GcdInt(d, x, y, a, b);
	x.Mul(x, a);
	y.Mul(y, b);
	x.Add(x, y);

	return CmpInt(x, d) == 0;
}


type gcdTest struct {
	a, b	int64;
	d, x, y	int64;
}


var gcdTests = []gcdTest{
	gcdTest{120, 23, 1, -9, 47},
}


func TestGcd(t *testing.T) {
	for i, test := range gcdTests {
		a := new(Int).New(test.a);
		b := new(Int).New(test.b);

		x := new(Int);
		y := new(Int);
		d := new(Int);

		expectedX := new(Int).New(test.x);
		expectedY := new(Int).New(test.y);
		expectedD := new(Int).New(test.d);

		GcdInt(d, x, y, a, b);

		if CmpInt(expectedX, x) != 0 ||
			CmpInt(expectedY, y) != 0 ||
			CmpInt(expectedD, d) != 0 {
			t.Errorf("#%d got (%s %s %s) want (%s %s %s)", i, x, y, d, expectedX, expectedY, expectedD);
		}
	}

	quick.Check(checkGcd, nil);
}
