// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "testing"

func TestCmpNN(t *testing.T) {
	// TODO(gri) write this test - all other tests depends on it
}


type funNN func(z, x, y []Word) []Word
type argNN struct {
	z, x, y []Word;
}


var sumNN = []argNN{
	argNN{},
	argNN{[]Word{1}, nil, []Word{1}},
	argNN{[]Word{1111111110}, []Word{123456789}, []Word{987654321}},
	argNN{[]Word{0, 0, 0, 1}, nil, []Word{0, 0, 0, 1}},
	argNN{[]Word{0, 0, 0, 1111111110}, []Word{0, 0, 0, 123456789}, []Word{0, 0, 0, 987654321}},
	argNN{[]Word{0, 0, 0, 1}, []Word{0, 0, _M}, []Word{0, 0, 1}},
}


var prodNN = []argNN{
	argNN{},
	argNN{nil, nil, nil},
	argNN{nil, []Word{991}, nil},
	argNN{[]Word{991}, []Word{991}, []Word{1}},
	argNN{[]Word{991 * 991}, []Word{991}, []Word{991}},
	argNN{[]Word{0, 0, 991 * 991}, []Word{0, 991}, []Word{0, 991}},
	argNN{[]Word{1 * 991, 2 * 991, 3 * 991, 4 * 991}, []Word{1, 2, 3, 4}, []Word{991}},
	argNN{[]Word{4, 11, 20, 30, 20, 11, 4}, []Word{1, 2, 3, 4}, []Word{4, 3, 2, 1}},
}


func TestSetN(t *testing.T) {
	for _, a := range sumNN {
		z := setN(nil, a.z);
		if cmpNN(z, a.z) != 0 {
			t.Errorf("got z = %v; want %v", z, a.z)
		}
	}
}


func testFunNN(t *testing.T, msg string, f funNN, a argNN) {
	z := f(nil, a.x, a.y);
	if cmpNN(z, a.z) != 0 {
		t.Errorf("%s%+v\n\tgot z = %v; want %v", msg, a, z, a.z)
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

	for _, a := range prodNN {
		arg := a;
		testFunNN(t, "mulNN", mulNN, arg);

		arg = argNN{a.z, a.y, a.x};
		testFunNN(t, "mulNN symmetric", mulNN, arg);
	}
}


type strN struct {
	x	[]Word;
	b	int;
	s	string;
}


var tabN = []strN{
	strN{nil, 10, "0"},
	strN{[]Word{1}, 10, "1"},
	strN{[]Word{10}, 10, "10"},
	strN{[]Word{1234567890}, 10, "1234567890"},
}


func TestStringN(t *testing.T) {
	for _, a := range tabN {
		s := stringN(a.x, a.b);
		if s != a.s {
			t.Errorf("stringN%+v\n\tgot s = %s; want %s", a, s, a.s)
		}

		x, b, n := scanN(nil, a.s, a.b);
		if cmpNN(x, a.x) != 0 {
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
	var x Word = 1 << (_W - 1);
	for i := 0; i <= int(_W); i++ {
		if leadingZeroBits(x) != i {
			t.Errorf("failed at %x: got %d want %d", x, leadingZeroBits(x), i)
		}
		x >>= 1;
	}
}


type shiftTest struct {
	in	[]Word;
	shift	int;
	out	[]Word;
}


var leftShiftTests = []shiftTest{
	shiftTest{nil, 0, nil},
	shiftTest{nil, 1, nil},
	shiftTest{[]Word{0}, 0, []Word{0}},
	shiftTest{[]Word{1}, 0, []Word{1}},
	shiftTest{[]Word{1}, 1, []Word{2}},
	shiftTest{[]Word{1 << (_W - 1)}, 1, []Word{0}},
	shiftTest{[]Word{1 << (_W - 1), 0}, 1, []Word{0, 1}},
}


func TestShiftLeft(t *testing.T) {
	for i, test := range leftShiftTests {
		dst := make([]Word, len(test.out));
		shiftLeft(dst, test.in, test.shift);
		for j, v := range dst {
			if test.out[j] != v {
				t.Errorf("#%d: got: %v want: %v", i, dst, test.out);
				break;
			}
		}
	}
}


var rightShiftTests = []shiftTest{
	shiftTest{nil, 0, nil},
	shiftTest{nil, 1, nil},
	shiftTest{[]Word{0}, 0, []Word{0}},
	shiftTest{[]Word{1}, 0, []Word{1}},
	shiftTest{[]Word{1}, 1, []Word{0}},
	shiftTest{[]Word{2}, 1, []Word{1}},
	shiftTest{[]Word{0, 1}, 1, []Word{1 << (_W - 1), 0}},
	shiftTest{[]Word{2, 1, 1}, 1, []Word{1<<(_W-1) + 1, 1 << (_W - 1), 0}},
}


func TestShiftRight(t *testing.T) {
	for i, test := range rightShiftTests {
		dst := make([]Word, len(test.out));
		shiftRight(dst, test.in, test.shift);
		for j, v := range dst {
			if test.out[j] != v {
				t.Errorf("#%d: got: %v want: %v", i, dst, test.out);
				break;
			}
		}
	}
}
