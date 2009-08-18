// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "testing"


func newZ(x int64) *Int {
	var z Int;
	return z.New(x);
}


type funZZ func(z, x, y *Int) *Int
type argZZ struct { z, x, y *Int }

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
	argZZ{newZ(-991*991), newZ(991), newZ(-991)},
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


var facts = map[int] string {
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
