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


func TestFunZZ(t *testing.T) {
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
