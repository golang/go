// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "testing"

func TestCmpNN(t *testing.T) {
	// TODO(gri) write this test - all other tests depends on it
}


type funNN func(z, x, y []Word) []Word
type argNN struct { z, x, y []Word }

var sumNN = []argNN{
	argNN{},
	argNN{[]Word{1}, nil, []Word{1}},
	argNN{[]Word{1111111110}, []Word{123456789}, []Word{987654321}},
	argNN{[]Word{0, 0, 0, 1}, nil, []Word{0, 0, 0, 1}},
	argNN{[]Word{0, 0, 0, 1111111110}, []Word{0, 0, 0, 123456789}, []Word{0, 0, 0, 987654321}},
	argNN{[]Word{0, 0, 0, 1}, []Word{0, 0, _M}, []Word{0, 0, 1}},
}


func TestSetN(t *testing.T) {
	for _, a := range sumNN {
		z := setN(nil, a.z);
		if cmpNN(z, a.z) != 0 {
			t.Errorf("got z = %v; want %v", z, a.z);
		}
	}
}


func testFunNN(t *testing.T, msg string, f funNN, a argNN) {
	z := f(nil, a.x, a.y);
	if cmpNN(z, a.z) != 0 {
		t.Errorf("%s%+v\n\tgot z = %v; want %v", msg, a, z, a.z);
	}
}


func TestFunNN(t *testing.T) {
	for _, a := range sumNN {
		arg := a;
		testFunNN(t, "addNN", addNN, arg);

		arg = argNN{a.z, a.y, a.x};
		testFunNN(t, "addNN symmetric", addNN, arg);

		arg = argNN{a.x, a.z, a.y};
		testFunNN(t, "subNN", subNN, arg);

		arg = argNN{a.y, a.z, a.x};
		testFunNN(t, "subNN symmetric", subNN, arg);
	}
}


type strN struct { x []Word; b int; s string }
var tabN = []strN{
	strN{nil, 10,  "0"},
	strN{[]Word{1}, 10, "1"},
	strN{[]Word{10}, 10, "10"},
	strN{[]Word{1234567890}, 10, "1234567890"},
}

func TestStringN(t *testing.T) {
	for _, a := range tabN {
		s := stringN(a.x, a.b);
		if s != a.s {
			t.Errorf("stringN%+v\n\tgot s = %s; want %s", a, s, a.s);
		}
	}
}
