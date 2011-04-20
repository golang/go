// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect_test

import (
	"bytes"
	"container/vector"
	"fmt"
	"io"
	"os"
	. "reflect"
	"runtime"
	"testing"
	"unsafe"
)

type integer int
type T struct {
	a int
	b float64
	c string
	d *int
}

type pair struct {
	i interface{}
	s string
}

func isDigit(c uint8) bool { return '0' <= c && c <= '9' }

func assert(t *testing.T, s, want string) {
	if s != want {
		t.Errorf("have %#q want %#q", s, want)
	}
}

func typestring(i interface{}) string { return Typeof(i).String() }

var typeTests = []pair{
	{struct{ x int }{}, "int"},
	{struct{ x int8 }{}, "int8"},
	{struct{ x int16 }{}, "int16"},
	{struct{ x int32 }{}, "int32"},
	{struct{ x int64 }{}, "int64"},
	{struct{ x uint }{}, "uint"},
	{struct{ x uint8 }{}, "uint8"},
	{struct{ x uint16 }{}, "uint16"},
	{struct{ x uint32 }{}, "uint32"},
	{struct{ x uint64 }{}, "uint64"},
	{struct{ x float32 }{}, "float32"},
	{struct{ x float64 }{}, "float64"},
	{struct{ x int8 }{}, "int8"},
	{struct{ x (**int8) }{}, "**int8"},
	{struct{ x (**integer) }{}, "**reflect_test.integer"},
	{struct{ x ([32]int32) }{}, "[32]int32"},
	{struct{ x ([]int8) }{}, "[]int8"},
	{struct{ x (map[string]int32) }{}, "map[string] int32"},
	{struct{ x (chan<- string) }{}, "chan<- string"},
	{struct {
		x struct {
			c chan *int32
			d float32
		}
	}{},
		"struct { c chan *int32; d float32 }",
	},
	{struct{ x (func(a int8, b int32)) }{}, "func(int8, int32)"},
	{struct {
		x struct {
			c func(chan *integer, *int8)
		}
	}{},
		"struct { c func(chan *reflect_test.integer, *int8) }",
	},
	{struct {
		x struct {
			a int8
			b int32
		}
	}{},
		"struct { a int8; b int32 }",
	},
	{struct {
		x struct {
			a int8
			b int8
			c int32
		}
	}{},
		"struct { a int8; b int8; c int32 }",
	},
	{struct {
		x struct {
			a int8
			b int8
			c int8
			d int32
		}
	}{},
		"struct { a int8; b int8; c int8; d int32 }",
	},
	{struct {
		x struct {
			a int8
			b int8
			c int8
			d int8
			e int32
		}
	}{},
		"struct { a int8; b int8; c int8; d int8; e int32 }",
	},
	{struct {
		x struct {
			a int8
			b int8
			c int8
			d int8
			e int8
			f int32
		}
	}{},
		"struct { a int8; b int8; c int8; d int8; e int8; f int32 }",
	},
	{struct {
		x struct {
			a int8 "hi there"
		}
	}{},
		`struct { a int8 "hi there" }`,
	},
	{struct {
		x struct {
			a int8 "hi \x00there\t\n\"\\"
		}
	}{},
		`struct { a int8 "hi \x00there\t\n\"\\" }`,
	},
	{struct {
		x struct {
			f func(args ...int)
		}
	}{},
		"struct { f func(...int) }",
	},
	{struct {
		x (interface {
			a(func(func(int) int) func(func(int)) int)
			b()
		})
	}{},
		"interface { a(func(func(int) int) func(func(int)) int); b() }",
	},
}

var valueTests = []pair{
	{new(int8), "8"},
	{new(int16), "16"},
	{new(int32), "32"},
	{new(int64), "64"},
	{new(uint8), "8"},
	{new(uint16), "16"},
	{new(uint32), "32"},
	{new(uint64), "64"},
	{new(float32), "256.25"},
	{new(float64), "512.125"},
	{new(string), "stringy cheese"},
	{new(bool), "true"},
	{new(*int8), "*int8(0)"},
	{new(**int8), "**int8(0)"},
	{new([5]int32), "[5]int32{0, 0, 0, 0, 0}"},
	{new(**integer), "**reflect_test.integer(0)"},
	{new(map[string]int32), "map[string] int32{<can't iterate on maps>}"},
	{new(chan<- string), "chan<- string"},
	{new(func(a int8, b int32)), "func(int8, int32)(0)"},
	{new(struct {
		c chan *int32
		d float32
	}),
		"struct { c chan *int32; d float32 }{chan *int32, 0}",
	},
	{new(struct{ c func(chan *integer, *int8) }),
		"struct { c func(chan *reflect_test.integer, *int8) }{func(chan *reflect_test.integer, *int8)(0)}",
	},
	{new(struct {
		a int8
		b int32
	}),
		"struct { a int8; b int32 }{0, 0}",
	},
	{new(struct {
		a int8
		b int8
		c int32
	}),
		"struct { a int8; b int8; c int32 }{0, 0, 0}",
	},
}

func testType(t *testing.T, i int, typ Type, want string) {
	s := typ.String()
	if s != want {
		t.Errorf("#%d: have %#q, want %#q", i, s, want)
	}
}

func TestTypes(t *testing.T) {
	for i, tt := range typeTests {
		testType(t, i, NewValue(tt.i).Field(0).Type(), tt.s)
	}
}

func TestSet(t *testing.T) {
	for i, tt := range valueTests {
		v := NewValue(tt.i).Elem()
		switch v.Kind() {
		case Int:
			v.SetInt(132)
		case Int8:
			v.SetInt(8)
		case Int16:
			v.SetInt(16)
		case Int32:
			v.SetInt(32)
		case Int64:
			v.SetInt(64)
		case Uint:
			v.SetUint(132)
		case Uint8:
			v.SetUint(8)
		case Uint16:
			v.SetUint(16)
		case Uint32:
			v.SetUint(32)
		case Uint64:
			v.SetUint(64)
		case Float32:
			v.SetFloat(256.25)
		case Float64:
			v.SetFloat(512.125)
		case Complex64:
			v.SetComplex(532.125 + 10i)
		case Complex128:
			v.SetComplex(564.25 + 1i)
		case String:
			v.SetString("stringy cheese")
		case Bool:
			v.SetBool(true)
		}
		s := valueToString(v)
		if s != tt.s {
			t.Errorf("#%d: have %#q, want %#q", i, s, tt.s)
		}
	}
}

func TestSetValue(t *testing.T) {
	for i, tt := range valueTests {
		v := NewValue(tt.i).Elem()
		switch v.Kind() {
		case Int:
			v.Set(NewValue(int(132)))
		case Int8:
			v.Set(NewValue(int8(8)))
		case Int16:
			v.Set(NewValue(int16(16)))
		case Int32:
			v.Set(NewValue(int32(32)))
		case Int64:
			v.Set(NewValue(int64(64)))
		case Uint:
			v.Set(NewValue(uint(132)))
		case Uint8:
			v.Set(NewValue(uint8(8)))
		case Uint16:
			v.Set(NewValue(uint16(16)))
		case Uint32:
			v.Set(NewValue(uint32(32)))
		case Uint64:
			v.Set(NewValue(uint64(64)))
		case Float32:
			v.Set(NewValue(float32(256.25)))
		case Float64:
			v.Set(NewValue(512.125))
		case Complex64:
			v.Set(NewValue(complex64(532.125 + 10i)))
		case Complex128:
			v.Set(NewValue(complex128(564.25 + 1i)))
		case String:
			v.Set(NewValue("stringy cheese"))
		case Bool:
			v.Set(NewValue(true))
		}
		s := valueToString(v)
		if s != tt.s {
			t.Errorf("#%d: have %#q, want %#q", i, s, tt.s)
		}
	}
}

var _i = 7

var valueToStringTests = []pair{
	{123, "123"},
	{123.5, "123.5"},
	{byte(123), "123"},
	{"abc", "abc"},
	{T{123, 456.75, "hello", &_i}, "reflect_test.T{123, 456.75, hello, *int(&7)}"},
	{new(chan *T), "*chan *reflect_test.T(&chan *reflect_test.T)"},
	{[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}"},
	{&[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "*[10]int(&[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})"},
	{[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}"},
	{&[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "*[]int(&[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})"},
}

func TestValueToString(t *testing.T) {
	for i, test := range valueToStringTests {
		s := valueToString(NewValue(test.i))
		if s != test.s {
			t.Errorf("#%d: have %#q, want %#q", i, s, test.s)
		}
	}
}

func TestArrayElemSet(t *testing.T) {
	v := NewValue(&[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}).Elem()
	v.Index(4).SetInt(123)
	s := valueToString(v)
	const want = "[10]int{1, 2, 3, 4, 123, 6, 7, 8, 9, 10}"
	if s != want {
		t.Errorf("[10]int: have %#q want %#q", s, want)
	}

	v = NewValue([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	v.Index(4).SetInt(123)
	s = valueToString(v)
	const want1 = "[]int{1, 2, 3, 4, 123, 6, 7, 8, 9, 10}"
	if s != want1 {
		t.Errorf("[]int: have %#q want %#q", s, want1)
	}
}

func TestPtrPointTo(t *testing.T) {
	var ip *int32
	var i int32 = 1234
	vip := NewValue(&ip)
	vi := NewValue(&i).Elem()
	vip.Elem().Set(vi.Addr())
	if *ip != 1234 {
		t.Errorf("got %d, want 1234", *ip)
	}

	ip = nil
	vp := NewValue(&ip).Elem()
	vp.Set(Zero(vp.Type()))
	if ip != nil {
		t.Errorf("got non-nil (%p), want nil", ip)
	}
}

func TestPtrSetNil(t *testing.T) {
	var i int32 = 1234
	ip := &i
	vip := NewValue(&ip)
	vip.Elem().Set(Zero(vip.Elem().Type()))
	if ip != nil {
		t.Errorf("got non-nil (%d), want nil", *ip)
	}
}

func TestMapSetNil(t *testing.T) {
	m := make(map[string]int)
	vm := NewValue(&m)
	vm.Elem().Set(Zero(vm.Elem().Type()))
	if m != nil {
		t.Errorf("got non-nil (%p), want nil", m)
	}
}


func TestAll(t *testing.T) {
	testType(t, 1, Typeof((int8)(0)), "int8")
	testType(t, 2, Typeof((*int8)(nil)).Elem(), "int8")

	typ := Typeof((*struct {
		c chan *int32
		d float32
	})(nil))
	testType(t, 3, typ, "*struct { c chan *int32; d float32 }")
	etyp := typ.Elem()
	testType(t, 4, etyp, "struct { c chan *int32; d float32 }")
	styp := etyp
	f := styp.Field(0)
	testType(t, 5, f.Type, "chan *int32")

	f, present := styp.FieldByName("d")
	if !present {
		t.Errorf("FieldByName says present field is absent")
	}
	testType(t, 6, f.Type, "float32")

	f, present = styp.FieldByName("absent")
	if present {
		t.Errorf("FieldByName says absent field is present")
	}

	typ = Typeof([32]int32{})
	testType(t, 7, typ, "[32]int32")
	testType(t, 8, typ.Elem(), "int32")

	typ = Typeof((map[string]*int32)(nil))
	testType(t, 9, typ, "map[string] *int32")
	mtyp := typ
	testType(t, 10, mtyp.Key(), "string")
	testType(t, 11, mtyp.Elem(), "*int32")

	typ = Typeof((chan<- string)(nil))
	testType(t, 12, typ, "chan<- string")
	testType(t, 13, typ.Elem(), "string")

	// make sure tag strings are not part of element type
	typ = Typeof(struct {
		d []uint32 "TAG"
	}{}).Field(0).Type
	testType(t, 14, typ, "[]uint32")
}

func TestInterfaceGet(t *testing.T) {
	var inter struct {
		E interface{}
	}
	inter.E = 123.456
	v1 := NewValue(&inter)
	v2 := v1.Elem().Field(0)
	assert(t, v2.Type().String(), "interface { }")
	i2 := v2.Interface()
	v3 := NewValue(i2)
	assert(t, v3.Type().String(), "float64")
}

func TestInterfaceValue(t *testing.T) {
	var inter struct {
		E interface{}
	}
	inter.E = 123.456
	v1 := NewValue(&inter)
	v2 := v1.Elem().Field(0)
	assert(t, v2.Type().String(), "interface { }")
	v3 := v2.Elem()
	assert(t, v3.Type().String(), "float64")

	i3 := v2.Interface()
	if _, ok := i3.(float64); !ok {
		t.Error("v2.Interface() did not return float64, got ", Typeof(i3))
	}
}

func TestFunctionValue(t *testing.T) {
	var x interface{} = func() {}
	v := NewValue(x)
	if v.Interface() != v.Interface() || v.Interface() != x {
		t.Fatalf("TestFunction != itself")
	}
	assert(t, v.Type().String(), "func()")
}

var appendTests = []struct {
	orig, extra []int
}{
	{make([]int, 2, 4), []int{22}},
	{make([]int, 2, 4), []int{22, 33, 44}},
}

func sameInts(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}
	for i, xx := range x {
		if xx != y[i] {
			return false
		}
	}
	return true
}

func TestAppend(t *testing.T) {
	for i, test := range appendTests {
		origLen, extraLen := len(test.orig), len(test.extra)
		want := append(test.orig, test.extra...)
		// Convert extra from []int to []Value.
		e0 := make([]Value, len(test.extra))
		for j, e := range test.extra {
			e0[j] = NewValue(e)
		}
		// Convert extra from []int to *SliceValue.
		e1 := NewValue(test.extra)
		// Test Append.
		a0 := NewValue(test.orig)
		have0 := Append(a0, e0...).Interface().([]int)
		if !sameInts(have0, want) {
			t.Errorf("Append #%d: have %v, want %v (%p %p)", i, have0, want, test.orig, have0)
		}
		// Check that the orig and extra slices were not modified.
		if len(test.orig) != origLen {
			t.Errorf("Append #%d origLen: have %v, want %v", i, len(test.orig), origLen)
		}
		if len(test.extra) != extraLen {
			t.Errorf("Append #%d extraLen: have %v, want %v", i, len(test.extra), extraLen)
		}
		// Test AppendSlice.
		a1 := NewValue(test.orig)
		have1 := AppendSlice(a1, e1).Interface().([]int)
		if !sameInts(have1, want) {
			t.Errorf("AppendSlice #%d: have %v, want %v", i, have1, want)
		}
		// Check that the orig and extra slices were not modified.
		if len(test.orig) != origLen {
			t.Errorf("AppendSlice #%d origLen: have %v, want %v", i, len(test.orig), origLen)
		}
		if len(test.extra) != extraLen {
			t.Errorf("AppendSlice #%d extraLen: have %v, want %v", i, len(test.extra), extraLen)
		}
	}
}

func TestCopy(t *testing.T) {
	a := []int{1, 2, 3, 4, 10, 9, 8, 7}
	b := []int{11, 22, 33, 44, 1010, 99, 88, 77, 66, 55, 44}
	c := []int{11, 22, 33, 44, 1010, 99, 88, 77, 66, 55, 44}
	for i := 0; i < len(b); i++ {
		if b[i] != c[i] {
			t.Fatalf("b != c before test")
		}
	}
	a1 := a
	b1 := b
	aa := NewValue(&a1).Elem()
	ab := NewValue(&b1).Elem()
	for tocopy := 1; tocopy <= 7; tocopy++ {
		aa.SetLen(tocopy)
		Copy(ab, aa)
		aa.SetLen(8)
		for i := 0; i < tocopy; i++ {
			if a[i] != b[i] {
				t.Errorf("(i) tocopy=%d a[%d]=%d, b[%d]=%d",
					tocopy, i, a[i], i, b[i])
			}
		}
		for i := tocopy; i < len(b); i++ {
			if b[i] != c[i] {
				if i < len(a) {
					t.Errorf("(ii) tocopy=%d a[%d]=%d, b[%d]=%d, c[%d]=%d",
						tocopy, i, a[i], i, b[i], i, c[i])
				} else {
					t.Errorf("(iii) tocopy=%d b[%d]=%d, c[%d]=%d",
						tocopy, i, b[i], i, c[i])
				}
			} else {
				t.Logf("tocopy=%d elem %d is okay\n", tocopy, i)
			}
		}
	}
}

func TestBigUnnamedStruct(t *testing.T) {
	b := struct{ a, b, c, d int64 }{1, 2, 3, 4}
	v := NewValue(b)
	b1 := v.Interface().(struct {
		a, b, c, d int64
	})
	if b1.a != b.a || b1.b != b.b || b1.c != b.c || b1.d != b.d {
		t.Errorf("NewValue(%v).Interface().(*Big) = %v", b, b1)
	}
}

type big struct {
	a, b, c, d, e int64
}

func TestBigStruct(t *testing.T) {
	b := big{1, 2, 3, 4, 5}
	v := NewValue(b)
	b1 := v.Interface().(big)
	if b1.a != b.a || b1.b != b.b || b1.c != b.c || b1.d != b.d || b1.e != b.e {
		t.Errorf("NewValue(%v).Interface().(big) = %v", b, b1)
	}
}

type Basic struct {
	x int
	y float32
}

type NotBasic Basic

type DeepEqualTest struct {
	a, b interface{}
	eq   bool
}

var deepEqualTests = []DeepEqualTest{
	// Equalities
	{1, 1, true},
	{int32(1), int32(1), true},
	{0.5, 0.5, true},
	{float32(0.5), float32(0.5), true},
	{"hello", "hello", true},
	{make([]int, 10), make([]int, 10), true},
	{&[3]int{1, 2, 3}, &[3]int{1, 2, 3}, true},
	{Basic{1, 0.5}, Basic{1, 0.5}, true},
	{os.Error(nil), os.Error(nil), true},
	{map[int]string{1: "one", 2: "two"}, map[int]string{2: "two", 1: "one"}, true},

	// Inequalities
	{1, 2, false},
	{int32(1), int32(2), false},
	{0.5, 0.6, false},
	{float32(0.5), float32(0.6), false},
	{"hello", "hey", false},
	{make([]int, 10), make([]int, 11), false},
	{&[3]int{1, 2, 3}, &[3]int{1, 2, 4}, false},
	{Basic{1, 0.5}, Basic{1, 0.6}, false},
	{Basic{1, 0}, Basic{2, 0}, false},
	{map[int]string{1: "one", 3: "two"}, map[int]string{2: "two", 1: "one"}, false},
	{map[int]string{1: "one", 2: "txo"}, map[int]string{2: "two", 1: "one"}, false},
	{map[int]string{1: "one"}, map[int]string{2: "two", 1: "one"}, false},
	{map[int]string{2: "two", 1: "one"}, map[int]string{1: "one"}, false},
	{nil, 1, false},
	{1, nil, false},

	// Mismatched types
	{1, 1.0, false},
	{int32(1), int64(1), false},
	{0.5, "hello", false},
	{[]int{1, 2, 3}, [3]int{1, 2, 3}, false},
	{&[3]interface{}{1, 2, 4}, &[3]interface{}{1, 2, "s"}, false},
	{Basic{1, 0.5}, NotBasic{1, 0.5}, false},
	{map[uint]string{1: "one", 2: "two"}, map[int]string{2: "two", 1: "one"}, false},
}

func TestDeepEqual(t *testing.T) {
	for _, test := range deepEqualTests {
		if r := DeepEqual(test.a, test.b); r != test.eq {
			t.Errorf("DeepEqual(%v, %v) = %v, want %v", test.a, test.b, r, test.eq)
		}
	}
}

func TestTypeof(t *testing.T) {
	for _, test := range deepEqualTests {
		v := NewValue(test.a)
		if !v.IsValid() {
			continue
		}
		typ := Typeof(test.a)
		if typ != v.Type() {
			t.Errorf("Typeof(%v) = %v, but NewValue(%v).Type() = %v", test.a, typ, test.a, v.Type())
		}
	}
}

type Recursive struct {
	x int
	r *Recursive
}

func TestDeepEqualRecursiveStruct(t *testing.T) {
	a, b := new(Recursive), new(Recursive)
	*a = Recursive{12, a}
	*b = Recursive{12, b}
	if !DeepEqual(a, b) {
		t.Error("DeepEqual(recursive same) = false, want true")
	}
}

type _Complex struct {
	a int
	b [3]*_Complex
	c *string
	d map[float64]float64
}

func TestDeepEqualComplexStruct(t *testing.T) {
	m := make(map[float64]float64)
	stra, strb := "hello", "hello"
	a, b := new(_Complex), new(_Complex)
	*a = _Complex{5, [3]*_Complex{a, b, a}, &stra, m}
	*b = _Complex{5, [3]*_Complex{b, a, a}, &strb, m}
	if !DeepEqual(a, b) {
		t.Error("DeepEqual(complex same) = false, want true")
	}
}

func TestDeepEqualComplexStructInequality(t *testing.T) {
	m := make(map[float64]float64)
	stra, strb := "hello", "helloo" // Difference is here
	a, b := new(_Complex), new(_Complex)
	*a = _Complex{5, [3]*_Complex{a, b, a}, &stra, m}
	*b = _Complex{5, [3]*_Complex{b, a, a}, &strb, m}
	if DeepEqual(a, b) {
		t.Error("DeepEqual(complex different) = true, want false")
	}
}


func check2ndField(x interface{}, offs uintptr, t *testing.T) {
	s := NewValue(x)
	f := s.Type().Field(1)
	if f.Offset != offs {
		t.Error("mismatched offsets in structure alignment:", f.Offset, offs)
	}
}

// Check that structure alignment & offsets viewed through reflect agree with those
// from the compiler itself.
func TestAlignment(t *testing.T) {
	type T1inner struct {
		a int
	}
	type T1 struct {
		T1inner
		f int
	}
	type T2inner struct {
		a, b int
	}
	type T2 struct {
		T2inner
		f int
	}

	x := T1{T1inner{2}, 17}
	check2ndField(x, uintptr(unsafe.Pointer(&x.f))-uintptr(unsafe.Pointer(&x)), t)

	x1 := T2{T2inner{2, 3}, 17}
	check2ndField(x1, uintptr(unsafe.Pointer(&x1.f))-uintptr(unsafe.Pointer(&x1)), t)
}

func Nil(a interface{}, t *testing.T) {
	n := NewValue(a).Field(0)
	if !n.IsNil() {
		t.Errorf("%v should be nil", a)
	}
}

func NotNil(a interface{}, t *testing.T) {
	n := NewValue(a).Field(0)
	if n.IsNil() {
		t.Errorf("value of type %v should not be nil", NewValue(a).Type().String())
	}
}

func TestIsNil(t *testing.T) {
	// These implement IsNil.
	// Wrap in extra struct to hide interface type.
	doNil := []interface{}{
		struct{ x *int }{},
		struct{ x interface{} }{},
		struct{ x map[string]int }{},
		struct{ x func() bool }{},
		struct{ x chan int }{},
		struct{ x []string }{},
	}
	for _, ts := range doNil {
		ty := Typeof(ts).Field(0).Type
		v := Zero(ty)
		v.IsNil() // panics if not okay to call
	}

	// Check the implementations
	var pi struct {
		x *int
	}
	Nil(pi, t)
	pi.x = new(int)
	NotNil(pi, t)

	var si struct {
		x []int
	}
	Nil(si, t)
	si.x = make([]int, 10)
	NotNil(si, t)

	var ci struct {
		x chan int
	}
	Nil(ci, t)
	ci.x = make(chan int)
	NotNil(ci, t)

	var mi struct {
		x map[int]int
	}
	Nil(mi, t)
	mi.x = make(map[int]int)
	NotNil(mi, t)

	var ii struct {
		x interface{}
	}
	Nil(ii, t)
	ii.x = 2
	NotNil(ii, t)

	var fi struct {
		x func(t *testing.T)
	}
	Nil(fi, t)
	fi.x = TestIsNil
	NotNil(fi, t)
}

func TestInterfaceExtraction(t *testing.T) {
	var s struct {
		w io.Writer
	}

	s.w = os.Stdout
	v := Indirect(NewValue(&s)).Field(0).Interface()
	if v != s.w.(interface{}) {
		t.Error("Interface() on interface: ", v, s.w)
	}
}

func TestNilPtrValueSub(t *testing.T) {
	var pi *int
	if pv := NewValue(pi); pv.Elem().IsValid() {
		t.Error("NewValue((*int)(nil)).Elem().IsValid()")
	}
}

func TestMap(t *testing.T) {
	m := map[string]int{"a": 1, "b": 2}
	mv := NewValue(m)
	if n := mv.Len(); n != len(m) {
		t.Errorf("Len = %d, want %d", n, len(m))
	}
	keys := mv.MapKeys()
	i := 0
	newmap := MakeMap(mv.Type())
	for k, v := range m {
		// Check that returned Keys match keys in range.
		// These aren't required to be in the same order,
		// but they are in this implementation, which makes
		// the test easier.
		if i >= len(keys) {
			t.Errorf("Missing key #%d %q", i, k)
		} else if kv := keys[i]; kv.String() != k {
			t.Errorf("Keys[%q] = %d, want %d", i, kv.Int(), k)
		}
		i++

		// Check that value lookup is correct.
		vv := mv.MapIndex(NewValue(k))
		if vi := vv.Int(); vi != int64(v) {
			t.Errorf("Key %q: have value %d, want %d", k, vi, v)
		}

		// Copy into new map.
		newmap.SetMapIndex(NewValue(k), NewValue(v))
	}
	vv := mv.MapIndex(NewValue("not-present"))
	if vv.IsValid() {
		t.Errorf("Invalid key: got non-nil value %s", valueToString(vv))
	}

	newm := newmap.Interface().(map[string]int)
	if len(newm) != len(m) {
		t.Errorf("length after copy: newm=%d, m=%d", newm, m)
	}

	for k, v := range newm {
		mv, ok := m[k]
		if mv != v {
			t.Errorf("newm[%q] = %d, but m[%q] = %d, %v", k, v, k, mv, ok)
		}
	}

	newmap.SetMapIndex(NewValue("a"), Value{})
	v, ok := newm["a"]
	if ok {
		t.Errorf("newm[\"a\"] = %d after delete", v)
	}

	mv = NewValue(&m).Elem()
	mv.Set(Zero(mv.Type()))
	if m != nil {
		t.Errorf("mv.Set(nil) failed")
	}
}

func TestChan(t *testing.T) {
	for loop := 0; loop < 2; loop++ {
		var c chan int
		var cv Value

		// check both ways to allocate channels
		switch loop {
		case 1:
			c = make(chan int, 1)
			cv = NewValue(c)
		case 0:
			cv = MakeChan(Typeof(c), 1)
			c = cv.Interface().(chan int)
		}

		// Send
		cv.Send(NewValue(2))
		if i := <-c; i != 2 {
			t.Errorf("reflect Send 2, native recv %d", i)
		}

		// Recv
		c <- 3
		if i, ok := cv.Recv(); i.Int() != 3 || !ok {
			t.Errorf("native send 3, reflect Recv %d, %t", i.Int(), ok)
		}

		// TryRecv fail
		val, ok := cv.TryRecv()
		if val.IsValid() || ok {
			t.Errorf("TryRecv on empty chan: %s, %t", valueToString(val), ok)
		}

		// TryRecv success
		c <- 4
		val, ok = cv.TryRecv()
		if !val.IsValid() {
			t.Errorf("TryRecv on ready chan got nil")
		} else if i := val.Int(); i != 4 || !ok {
			t.Errorf("native send 4, TryRecv %d, %t", i, ok)
		}

		// TrySend fail
		c <- 100
		ok = cv.TrySend(NewValue(5))
		i := <-c
		if ok {
			t.Errorf("TrySend on full chan succeeded: value %d", i)
		}

		// TrySend success
		ok = cv.TrySend(NewValue(6))
		if !ok {
			t.Errorf("TrySend on empty chan failed")
		} else {
			if i = <-c; i != 6 {
				t.Errorf("TrySend 6, recv %d", i)
			}
		}

		// Close
		c <- 123
		cv.Close()
		if i, ok := cv.Recv(); i.Int() != 123 || !ok {
			t.Errorf("send 123 then close; Recv %d, %t", i.Int(), ok)
		}
		if i, ok := cv.Recv(); i.Int() != 0 || ok {
			t.Errorf("after close Recv %d, %t", i.Int(), ok)
		}
	}

	// check creation of unbuffered channel
	var c chan int
	cv := MakeChan(Typeof(c), 0)
	c = cv.Interface().(chan int)
	if cv.TrySend(NewValue(7)) {
		t.Errorf("TrySend on sync chan succeeded")
	}
	if v, ok := cv.TryRecv(); v.IsValid() || ok {
		t.Errorf("TryRecv on sync chan succeeded: isvalid=%v ok=%v", v.IsValid(), ok)
	}

	// len/cap
	cv = MakeChan(Typeof(c), 10)
	c = cv.Interface().(chan int)
	for i := 0; i < 3; i++ {
		c <- i
	}
	if l, m := cv.Len(), cv.Cap(); l != len(c) || m != cap(c) {
		t.Errorf("Len/Cap = %d/%d want %d/%d", l, m, len(c), cap(c))
	}

}

// Difficult test for function call because of
// implicit padding between arguments.
func dummy(b byte, c int, d byte) (i byte, j int, k byte) {
	return b, c, d
}

func TestFunc(t *testing.T) {
	ret := NewValue(dummy).Call([]Value{NewValue(byte(10)), NewValue(20), NewValue(byte(30))})
	if len(ret) != 3 {
		t.Fatalf("Call returned %d values, want 3", len(ret))
	}

	i := byte(ret[0].Uint())
	j := int(ret[1].Int())
	k := byte(ret[2].Uint())
	if i != 10 || j != 20 || k != 30 {
		t.Errorf("Call returned %d, %d, %d; want 10, 20, 30", i, j, k)
	}
}

type Point struct {
	x, y int
}

func (p Point) Dist(scale int) int {
	//	println("Point.Dist", p.x, p.y, scale)
	return p.x*p.x*scale + p.y*p.y*scale
}

func TestMethod(t *testing.T) {
	// Non-curried method of type.
	p := Point{3, 4}
	i := Typeof(p).Method(0).Func.Call([]Value{NewValue(p), NewValue(10)})[0].Int()
	if i != 250 {
		t.Errorf("Type Method returned %d; want 250", i)
	}

	i = Typeof(&p).Method(0).Func.Call([]Value{NewValue(&p), NewValue(10)})[0].Int()
	if i != 250 {
		t.Errorf("Pointer Type Method returned %d; want 250", i)
	}

	// Curried method of value.
	i = NewValue(p).Method(0).Call([]Value{NewValue(10)})[0].Int()
	if i != 250 {
		t.Errorf("Value Method returned %d; want 250", i)
	}

	// Curried method of pointer.
	i = NewValue(&p).Method(0).Call([]Value{NewValue(10)})[0].Int()
	if i != 250 {
		t.Errorf("Value Method returned %d; want 250", i)
	}

	// Curried method of interface value.
	// Have to wrap interface value in a struct to get at it.
	// Passing it to NewValue directly would
	// access the underlying Point, not the interface.
	var s = struct {
		X interface {
			Dist(int) int
		}
	}{p}
	pv := NewValue(s).Field(0)
	i = pv.Method(0).Call([]Value{NewValue(10)})[0].Int()
	if i != 250 {
		t.Errorf("Interface Method returned %d; want 250", i)
	}
}

func TestInterfaceSet(t *testing.T) {
	p := &Point{3, 4}

	var s struct {
		I interface{}
		P interface {
			Dist(int) int
		}
	}
	sv := NewValue(&s).Elem()
	sv.Field(0).Set(NewValue(p))
	if q := s.I.(*Point); q != p {
		t.Errorf("i: have %p want %p", q, p)
	}

	pv := sv.Field(1)
	pv.Set(NewValue(p))
	if q := s.P.(*Point); q != p {
		t.Errorf("i: have %p want %p", q, p)
	}

	i := pv.Method(0).Call([]Value{NewValue(10)})[0].Int()
	if i != 250 {
		t.Errorf("Interface Method returned %d; want 250", i)
	}
}

type T1 struct {
	a string
	int
}

func TestAnonymousFields(t *testing.T) {
	var field StructField
	var ok bool
	var t1 T1
	type1 := Typeof(t1)
	if field, ok = type1.FieldByName("int"); !ok {
		t.Error("no field 'int'")
	}
	if field.Index[0] != 1 {
		t.Error("field index should be 1; is", field.Index)
	}
}

type FTest struct {
	s     interface{}
	name  string
	index []int
	value int
}

type D1 struct {
	d int
}
type D2 struct {
	d int
}

type S0 struct {
	a, b, c int
	D1
	D2
}

type S1 struct {
	b int
	S0
}

type S2 struct {
	a int
	*S1
}

type S1x struct {
	S1
}

type S1y struct {
	S1
}

type S3 struct {
	S1x
	S2
	d, e int
	*S1y
}

type S4 struct {
	*S4
	a int
}

var fieldTests = []FTest{
	{struct{}{}, "", nil, 0},
	{struct{}{}, "foo", nil, 0},
	{S0{a: 'a'}, "a", []int{0}, 'a'},
	{S0{}, "d", nil, 0},
	{S1{S0: S0{a: 'a'}}, "a", []int{1, 0}, 'a'},
	{S1{b: 'b'}, "b", []int{0}, 'b'},
	{S1{}, "S0", []int{1}, 0},
	{S1{S0: S0{c: 'c'}}, "c", []int{1, 2}, 'c'},
	{S2{a: 'a'}, "a", []int{0}, 'a'},
	{S2{}, "S1", []int{1}, 0},
	{S2{S1: &S1{b: 'b'}}, "b", []int{1, 0}, 'b'},
	{S2{S1: &S1{S0: S0{c: 'c'}}}, "c", []int{1, 1, 2}, 'c'},
	{S2{}, "d", nil, 0},
	{S3{}, "S1", nil, 0},
	{S3{S2: S2{a: 'a'}}, "a", []int{1, 0}, 'a'},
	{S3{}, "b", nil, 0},
	{S3{d: 'd'}, "d", []int{2}, 0},
	{S3{e: 'e'}, "e", []int{3}, 'e'},
	{S4{a: 'a'}, "a", []int{1}, 'a'},
	{S4{}, "b", nil, 0},
}

func TestFieldByIndex(t *testing.T) {
	for _, test := range fieldTests {
		s := Typeof(test.s)
		f := s.FieldByIndex(test.index)
		if f.Name != "" {
			if test.index != nil {
				if f.Name != test.name {
					t.Errorf("%s.%s found; want %s", s.Name(), f.Name, test.name)
				}
			} else {
				t.Errorf("%s.%s found", s.Name(), f.Name)
			}
		} else if len(test.index) > 0 {
			t.Errorf("%s.%s not found", s.Name(), test.name)
		}

		if test.value != 0 {
			v := NewValue(test.s).FieldByIndex(test.index)
			if v.IsValid() {
				if x, ok := v.Interface().(int); ok {
					if x != test.value {
						t.Errorf("%s%v is %d; want %d", s.Name(), test.index, x, test.value)
					}
				} else {
					t.Errorf("%s%v value not an int", s.Name(), test.index)
				}
			} else {
				t.Errorf("%s%v value not found", s.Name(), test.index)
			}
		}
	}
}

func TestFieldByName(t *testing.T) {
	for _, test := range fieldTests {
		s := Typeof(test.s)
		f, found := s.FieldByName(test.name)
		if found {
			if test.index != nil {
				// Verify field depth and index.
				if len(f.Index) != len(test.index) {
					t.Errorf("%s.%s depth %d; want %d", s.Name(), test.name, len(f.Index), len(test.index))
				} else {
					for i, x := range f.Index {
						if x != test.index[i] {
							t.Errorf("%s.%s.Index[%d] is %d; want %d", s.Name(), test.name, i, x, test.index[i])
						}
					}
				}
			} else {
				t.Errorf("%s.%s found", s.Name(), f.Name)
			}
		} else if len(test.index) > 0 {
			t.Errorf("%s.%s not found", s.Name(), test.name)
		}

		if test.value != 0 {
			v := NewValue(test.s).FieldByName(test.name)
			if v.IsValid() {
				if x, ok := v.Interface().(int); ok {
					if x != test.value {
						t.Errorf("%s.%s is %d; want %d", s.Name(), test.name, x, test.value)
					}
				} else {
					t.Errorf("%s.%s value not an int", s.Name(), test.name)
				}
			} else {
				t.Errorf("%s.%s value not found", s.Name(), test.name)
			}
		}
	}
}

func TestImportPath(t *testing.T) {
	if path := Typeof(vector.Vector{}).PkgPath(); path != "container/vector" {
		t.Errorf("Typeof(vector.Vector{}).PkgPath() = %q, want \"container/vector\"", path)
	}
}

func TestDotDotDot(t *testing.T) {
	// Test example from FuncType.DotDotDot documentation.
	var f func(x int, y ...float64)
	typ := Typeof(f)
	if typ.NumIn() == 2 && typ.In(0) == Typeof(int(0)) {
		sl := typ.In(1)
		if sl.Kind() == Slice {
			if sl.Elem() == Typeof(0.0) {
				// ok
				return
			}
		}
	}

	// Failed
	t.Errorf("want NumIn() = 2, In(0) = int, In(1) = []float64")
	s := fmt.Sprintf("have NumIn() = %d", typ.NumIn())
	for i := 0; i < typ.NumIn(); i++ {
		s += fmt.Sprintf(", In(%d) = %s", i, typ.In(i))
	}
	t.Error(s)
}

type inner struct {
	x int
}

type outer struct {
	y int
	inner
}

func (*inner) m() {}
func (*outer) m() {}

func TestNestedMethods(t *testing.T) {
	typ := Typeof((*outer)(nil))
	if typ.NumMethod() != 1 || typ.Method(0).Func.Pointer() != NewValue((*outer).m).Pointer() {
		t.Errorf("Wrong method table for outer: (m=%p)", (*outer).m)
		for i := 0; i < typ.NumMethod(); i++ {
			m := typ.Method(i)
			t.Errorf("\t%d: %s %#x\n", i, m.Name, m.Func.Pointer())
		}
	}
}

type innerInt struct {
	x int
}

type outerInt struct {
	y int
	innerInt
}

func (i *innerInt) m() int {
	return i.x
}

func TestEmbeddedMethods(t *testing.T) {
	typ := Typeof((*outerInt)(nil))
	if typ.NumMethod() != 1 || typ.Method(0).Func.Pointer() != NewValue((*outerInt).m).Pointer() {
		t.Errorf("Wrong method table for outerInt: (m=%p)", (*outerInt).m)
		for i := 0; i < typ.NumMethod(); i++ {
			m := typ.Method(i)
			t.Errorf("\t%d: %s %#x\n", i, m.Name, m.Func.Pointer())
		}
	}

	i := &innerInt{3}
	if v := NewValue(i).Method(0).Call(nil)[0].Int(); v != 3 {
		t.Errorf("i.m() = %d, want 3", v)
	}

	o := &outerInt{1, innerInt{2}}
	if v := NewValue(o).Method(0).Call(nil)[0].Int(); v != 2 {
		t.Errorf("i.m() = %d, want 2", v)
	}

	f := (*outerInt).m
	if v := f(o); v != 2 {
		t.Errorf("f(o) = %d, want 2", v)
	}
}

func TestPtrTo(t *testing.T) {
	var i int

	typ := Typeof(i)
	for i = 0; i < 100; i++ {
		typ = PtrTo(typ)
	}
	for i = 0; i < 100; i++ {
		typ = typ.Elem()
	}
	if typ != Typeof(i) {
		t.Errorf("after 100 PtrTo and Elem, have %s, want %s", typ, Typeof(i))
	}
}

func TestAddr(t *testing.T) {
	var p struct {
		X, Y int
	}

	v := NewValue(&p)
	v = v.Elem()
	v = v.Addr()
	v = v.Elem()
	v = v.Field(0)
	v.SetInt(2)
	if p.X != 2 {
		t.Errorf("Addr.Elem.Set failed to set value")
	}

	// Again but take address of the NewValue value.
	// Exercises generation of PtrTypes not present in the binary.
	q := &p
	v = NewValue(&q).Elem()
	v = v.Addr()
	v = v.Elem()
	v = v.Elem()
	v = v.Addr()
	v = v.Elem()
	v = v.Field(0)
	v.SetInt(3)
	if p.X != 3 {
		t.Errorf("Addr.Elem.Set failed to set value")
	}

	// Starting without pointer we should get changed value
	// in interface.
	qq := p
	v = NewValue(&qq).Elem()
	v0 := v
	v = v.Addr()
	v = v.Elem()
	v = v.Field(0)
	v.SetInt(4)
	if p.X != 3 { // should be unchanged from last time
		t.Errorf("somehow value Set changed original p")
	}
	p = v0.Interface().(struct {
		X, Y int
	})
	if p.X != 4 {
		t.Errorf("Addr.Elem.Set valued to set value in top value")
	}
}

func noAlloc(t *testing.T, n int, f func(int)) {
	// once to prime everything
	f(-1)
	runtime.MemStats.Mallocs = 0

	for j := 0; j < n; j++ {
		f(j)
	}
	if runtime.MemStats.Mallocs != 0 {
		t.Fatalf("%d mallocs after %d iterations", runtime.MemStats.Mallocs, n)
	}
}

func TestAllocations(t *testing.T) {
	noAlloc(t, 100, func(j int) {
		var i interface{}
		var v Value
		i = 42 + j
		v = NewValue(i)
		if int(v.Int()) != 42+j {
			panic("wrong int")
		}
	})
}

func TestSmallNegativeInt(t *testing.T) {
	i := int16(-1)
	v := NewValue(i)
	if v.Int() != -1 {
		t.Errorf("int16(-1).Int() returned %v", v.Int())
	}
}

func TestSlice(t *testing.T) {
	xs := []int{1, 2, 3, 4, 5, 6, 7, 8}
	v := NewValue(xs).Slice(3, 5).Interface().([]int)
	if len(v) != 2 || v[0] != 4 || v[1] != 5 {
		t.Errorf("xs.Slice(3, 5) = %v", v)
	}

	xa := [7]int{10, 20, 30, 40, 50, 60, 70}
	v = NewValue(&xa).Elem().Slice(2, 5).Interface().([]int)
	if len(v) != 3 || v[0] != 30 || v[1] != 40 || v[2] != 50 {
		t.Errorf("xa.Slice(2, 5) = %v", v)
	}
}

func TestVariadic(t *testing.T) {
	var b bytes.Buffer
	V := NewValue

	b.Reset()
	V(fmt.Fprintf).Call([]Value{V(&b), V("%s, %d world"), V("hello"), V(42)})
	if b.String() != "hello, 42 world" {
		t.Errorf("after Fprintf Call: %q != %q", b.String(), "hello 42 world")
	}

	b.Reset()
	V(fmt.Fprintf).CallSlice([]Value{V(&b), V("%s, %d world"), V([]interface{}{"hello", 42})})
	if b.String() != "hello, 42 world" {
		t.Errorf("after Fprintf CallSlice: %q != %q", b.String(), "hello 42 world")
	}
}
