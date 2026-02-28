// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests of generated equality functions.

package test

import (
	"reflect"
	"testing"
	"unsafe"
)

//go:noinline
func checkEq(t *testing.T, x, y any) {
	// Make sure we don't inline the equality test.
	if x != y {
		t.Errorf("%#v != %#v, wanted equal", x, y)
	}
}

//go:noinline
func checkNe(t *testing.T, x, y any) {
	// Make sure we don't inline the equality test.
	if x == y {
		t.Errorf("%#v == %#v, wanted not equal", x, y)
	}
}

//go:noinline
func checkPanic(t *testing.T, x, y any) {
	defer func() {
		if recover() == nil {
			t.Errorf("%#v == %#v didn't panic", x, y)
		}
	}()
	_ = x == y
}

type fooComparable struct {
	x int
}

func (f fooComparable) foo() {
}

type fooIncomparable struct {
	b func()
}

func (i fooIncomparable) foo() {
}

type eqResult int

const (
	eq eqResult = iota
	ne
	panic_
)

func (x eqResult) String() string {
	return []string{eq: "eq", ne: "ne", panic_: "panic"}[x]
}

// testEq returns eq if x==y, ne if x!=y, or panic_ if the comparison panics.
func testEq(x, y any) (r eqResult) {
	defer func() {
		if e := recover(); e != nil {
			r = panic_
		}
	}()
	r = ne
	if x == y {
		r = eq
	}
	return
}

// testCompare make two instances of struct type typ, then
// assigns its len(vals) fields one value from each slice in vals.
// Then it checks the results against a "manual" comparison field
// by field.
func testCompare(t *testing.T, typ reflect.Type, vals [][]any) {
	if len(vals) != typ.NumField() {
		t.Fatalf("bad test, have %d fields in the list, but %d fields in the type", len(vals), typ.NumField())
	}

	x := reflect.New(typ).Elem()
	y := reflect.New(typ).Elem()
	ps := powerSet(vals)    // all possible settings of fields of the test type.
	for _, xf := range ps { // Pick fields for x
		for _, yf := range ps { // Pick fields for y
			// Make x and y from their chosen fields.
			for i, f := range xf {
				x.Field(i).Set(reflect.ValueOf(f))
			}
			for i, f := range yf {
				y.Field(i).Set(reflect.ValueOf(f))
			}
			// Compute what we want the result to be.
			want := eq
			for i := range len(vals) {
				if c := testEq(xf[i], yf[i]); c != eq {
					want = c
					break
				}
			}
			// Compute actual result using generated equality function.
			got := testEq(x.Interface(), y.Interface())
			if got != want {
				t.Errorf("%#v == %#v, got %s want %s\n", x, y, got, want)
			}
		}
	}
}

// powerset returns all possible sequences of choosing one
// element from each entry in s.
// For instance, if s = {{1,2}, {a,b}}, then
// it returns {{1,a},{1,b},{2,a},{2,b}}.
func powerSet(s [][]any) [][]any {
	if len(s) == 0 {
		return [][]any{{}}
	}
	p := powerSet(s[:len(s)-1]) // powerset from first len(s)-1 entries
	var r [][]any
	for _, head := range p {
		// add one more entry.
		for _, v := range s[len(s)-1] {
			x := make([]any, 0, len(s))
			x = append(x, head...)
			x = append(x, v)
			r = append(r, x)
		}
	}
	return r
}

func TestCompareKinds1(t *testing.T) {
	type S struct {
		X0 int8
		X1 int16
		X2 int32
		X3 int64
		X4 float32
		X5 float64
	}
	testCompare(t, reflect.TypeOf(S{}), [][]any{
		{int8(0), int8(1)},
		{int16(0), int16(1), int16(1 << 14)},
		{int32(0), int32(1), int32(1 << 30)},
		{int64(0), int64(1), int64(1 << 62)},
		{float32(0), float32(1.0)},
		{0.0, 1.0},
	})
}
func TestCompareKinds2(t *testing.T) {
	type S struct {
		X0 uint8
		X1 uint16
		X2 uint32
		X3 uint64
		X4 uintptr
		X5 bool
	}
	testCompare(t, reflect.TypeOf(S{}), [][]any{
		{uint8(0), uint8(1)},
		{uint16(0), uint16(1), uint16(1 << 15)},
		{uint32(0), uint32(1), uint32(1 << 31)},
		{uint64(0), uint64(1), uint64(1 << 63)},
		{uintptr(0), uintptr(1)},
		{false, true},
	})
}
func TestCompareKinds3(t *testing.T) {
	type S struct {
		X0 complex64
		X1 complex128
		X2 *byte
		X3 chan int
		X4 unsafe.Pointer
	}
	testCompare(t, reflect.TypeOf(S{}), [][]any{
		{complex64(1 + 1i), complex64(1 + 2i), complex64(2 + 1i)},
		{complex128(1 + 1i), complex128(1 + 2i), complex128(2 + 1i)},
		{new(byte), new(byte)},
		{make(chan int), make(chan int)},
		{unsafe.Pointer(new(byte)), unsafe.Pointer(new(byte))},
	})
}

func TestCompareOrdering(t *testing.T) {
	type S struct {
		A string
		E any
		B string
	}

	testCompare(t, reflect.TypeOf(S{}), [][]any{
		{"a", "b", "cc"},
		{3, []byte{0}, []byte{1}},
		{"a", "b", "cc"},
	})
}
func TestCompareInterfaces(t *testing.T) {
	type S struct {
		A any
		B fooer
	}
	testCompare(t, reflect.TypeOf(S{}), [][]any{
		{3, []byte{0}},
		{fooComparable{x: 3}, fooIncomparable{b: nil}},
	})
}

func TestCompareSkip(t *testing.T) {
	type S struct {
		A int8
		B int16
	}
	type S2 struct {
		A       int8
		padding int8
		B       int16
	}
	x := S{A: 1, B: 3}
	y := S{A: 1, B: 3}
	(*S2)(unsafe.Pointer(&x)).padding = 88
	(*S2)(unsafe.Pointer(&y)).padding = 99

	want := eq
	if got := testEq(x, y); got != want {
		t.Errorf("%#v == %#v, got %s want %s", x, y, got, want)
	}
}

func TestCompareMemequal(t *testing.T) {
	type S struct {
		s1 string
		d  [100]byte
		s2 string
	}
	var x, y S

	checkEq(t, x, y)
	y.d[0] = 1
	checkNe(t, x, y)
	y.d[0] = 0
	y.d[99] = 1
	checkNe(t, x, y)
}

func TestComparePanic(t *testing.T) {
	type S struct {
		X0 string
		X1 any
		X2 string
		X3 fooer
		X4 string
	}
	testCompare(t, reflect.TypeOf(S{}), [][]any{
		{"a", "b", "cc"}, // length equal, as well as length unequal
		{3, []byte{1}},   // comparable and incomparable
		{"a", "b", "cc"}, // length equal, as well as length unequal
		{fooComparable{x: 3}, fooIncomparable{b: nil}}, // comparable and incomparable
		{"a", "b", "cc"}, // length equal, as well as length unequal
	})
}

func TestCompareArray(t *testing.T) {
	type S struct {
		X0 string
		X1 [100]string
		X2 string
	}
	x := S{X0: "a", X2: "b"}
	y := x
	checkEq(t, x, y)
	x.X0 = "c"
	checkNe(t, x, y)
	x.X0 = "a"
	x.X2 = "c"
	checkNe(t, x, y)
	x.X2 = "b"
	checkEq(t, x, y)

	for i := 0; i < 100; i++ {
		x.X1[i] = "d"
		checkNe(t, x, y)
		y.X1[i] = "e"
		checkNe(t, x, y)
		x.X1[i] = ""
		y.X1[i] = ""
		checkEq(t, x, y)
	}
}
