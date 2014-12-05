// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect_test

import (
	"bytes"
	"encoding/base64"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	. "reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"
)

func TestBool(t *testing.T) {
	v := ValueOf(true)
	if v.Bool() != true {
		t.Fatal("ValueOf(true).Bool() = false")
	}
}

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

func typestring(i interface{}) string { return TypeOf(i).String() }

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
	{struct{ x (map[string]int32) }{}, "map[string]int32"},
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
			a int8 `reflect:"hi there"`
		}
	}{},
		`struct { a int8 "reflect:\"hi there\"" }`,
	},
	{struct {
		x struct {
			a int8 `reflect:"hi \x00there\t\n\"\\"`
		}
	}{},
		`struct { a int8 "reflect:\"hi \\x00there\\t\\n\\\"\\\\\"" }`,
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
		"interface { reflect_test.a(func(func(int) int) func(func(int)) int); reflect_test.b() }",
	},
}

var valueTests = []pair{
	{new(int), "132"},
	{new(int8), "8"},
	{new(int16), "16"},
	{new(int32), "32"},
	{new(int64), "64"},
	{new(uint), "132"},
	{new(uint8), "8"},
	{new(uint16), "16"},
	{new(uint32), "32"},
	{new(uint64), "64"},
	{new(float32), "256.25"},
	{new(float64), "512.125"},
	{new(complex64), "532.125+10i"},
	{new(complex128), "564.25+1i"},
	{new(string), "stringy cheese"},
	{new(bool), "true"},
	{new(*int8), "*int8(0)"},
	{new(**int8), "**int8(0)"},
	{new([5]int32), "[5]int32{0, 0, 0, 0, 0}"},
	{new(**integer), "**reflect_test.integer(0)"},
	{new(map[string]int32), "map[string]int32{<can't iterate on maps>}"},
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
		testType(t, i, ValueOf(tt.i).Field(0).Type(), tt.s)
	}
}

func TestSet(t *testing.T) {
	for i, tt := range valueTests {
		v := ValueOf(tt.i)
		v = v.Elem()
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
		v := ValueOf(tt.i).Elem()
		switch v.Kind() {
		case Int:
			v.Set(ValueOf(int(132)))
		case Int8:
			v.Set(ValueOf(int8(8)))
		case Int16:
			v.Set(ValueOf(int16(16)))
		case Int32:
			v.Set(ValueOf(int32(32)))
		case Int64:
			v.Set(ValueOf(int64(64)))
		case Uint:
			v.Set(ValueOf(uint(132)))
		case Uint8:
			v.Set(ValueOf(uint8(8)))
		case Uint16:
			v.Set(ValueOf(uint16(16)))
		case Uint32:
			v.Set(ValueOf(uint32(32)))
		case Uint64:
			v.Set(ValueOf(uint64(64)))
		case Float32:
			v.Set(ValueOf(float32(256.25)))
		case Float64:
			v.Set(ValueOf(512.125))
		case Complex64:
			v.Set(ValueOf(complex64(532.125 + 10i)))
		case Complex128:
			v.Set(ValueOf(complex128(564.25 + 1i)))
		case String:
			v.Set(ValueOf("stringy cheese"))
		case Bool:
			v.Set(ValueOf(true))
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
		s := valueToString(ValueOf(test.i))
		if s != test.s {
			t.Errorf("#%d: have %#q, want %#q", i, s, test.s)
		}
	}
}

func TestArrayElemSet(t *testing.T) {
	v := ValueOf(&[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}).Elem()
	v.Index(4).SetInt(123)
	s := valueToString(v)
	const want = "[10]int{1, 2, 3, 4, 123, 6, 7, 8, 9, 10}"
	if s != want {
		t.Errorf("[10]int: have %#q want %#q", s, want)
	}

	v = ValueOf([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
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
	vip := ValueOf(&ip)
	vi := ValueOf(&i).Elem()
	vip.Elem().Set(vi.Addr())
	if *ip != 1234 {
		t.Errorf("got %d, want 1234", *ip)
	}

	ip = nil
	vp := ValueOf(&ip).Elem()
	vp.Set(Zero(vp.Type()))
	if ip != nil {
		t.Errorf("got non-nil (%p), want nil", ip)
	}
}

func TestPtrSetNil(t *testing.T) {
	var i int32 = 1234
	ip := &i
	vip := ValueOf(&ip)
	vip.Elem().Set(Zero(vip.Elem().Type()))
	if ip != nil {
		t.Errorf("got non-nil (%d), want nil", *ip)
	}
}

func TestMapSetNil(t *testing.T) {
	m := make(map[string]int)
	vm := ValueOf(&m)
	vm.Elem().Set(Zero(vm.Elem().Type()))
	if m != nil {
		t.Errorf("got non-nil (%p), want nil", m)
	}
}

func TestAll(t *testing.T) {
	testType(t, 1, TypeOf((int8)(0)), "int8")
	testType(t, 2, TypeOf((*int8)(nil)).Elem(), "int8")

	typ := TypeOf((*struct {
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

	typ = TypeOf([32]int32{})
	testType(t, 7, typ, "[32]int32")
	testType(t, 8, typ.Elem(), "int32")

	typ = TypeOf((map[string]*int32)(nil))
	testType(t, 9, typ, "map[string]*int32")
	mtyp := typ
	testType(t, 10, mtyp.Key(), "string")
	testType(t, 11, mtyp.Elem(), "*int32")

	typ = TypeOf((chan<- string)(nil))
	testType(t, 12, typ, "chan<- string")
	testType(t, 13, typ.Elem(), "string")

	// make sure tag strings are not part of element type
	typ = TypeOf(struct {
		d []uint32 `reflect:"TAG"`
	}{}).Field(0).Type
	testType(t, 14, typ, "[]uint32")
}

func TestInterfaceGet(t *testing.T) {
	var inter struct {
		E interface{}
	}
	inter.E = 123.456
	v1 := ValueOf(&inter)
	v2 := v1.Elem().Field(0)
	assert(t, v2.Type().String(), "interface {}")
	i2 := v2.Interface()
	v3 := ValueOf(i2)
	assert(t, v3.Type().String(), "float64")
}

func TestInterfaceValue(t *testing.T) {
	var inter struct {
		E interface{}
	}
	inter.E = 123.456
	v1 := ValueOf(&inter)
	v2 := v1.Elem().Field(0)
	assert(t, v2.Type().String(), "interface {}")
	v3 := v2.Elem()
	assert(t, v3.Type().String(), "float64")

	i3 := v2.Interface()
	if _, ok := i3.(float64); !ok {
		t.Error("v2.Interface() did not return float64, got ", TypeOf(i3))
	}
}

func TestFunctionValue(t *testing.T) {
	var x interface{} = func() {}
	v := ValueOf(x)
	if fmt.Sprint(v.Interface()) != fmt.Sprint(x) {
		t.Fatalf("TestFunction returned wrong pointer")
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
			e0[j] = ValueOf(e)
		}
		// Convert extra from []int to *SliceValue.
		e1 := ValueOf(test.extra)
		// Test Append.
		a0 := ValueOf(test.orig)
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
		a1 := ValueOf(test.orig)
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
	aa := ValueOf(&a1).Elem()
	ab := ValueOf(&b1).Elem()
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

func TestCopyArray(t *testing.T) {
	a := [8]int{1, 2, 3, 4, 10, 9, 8, 7}
	b := [11]int{11, 22, 33, 44, 1010, 99, 88, 77, 66, 55, 44}
	c := b
	aa := ValueOf(&a).Elem()
	ab := ValueOf(&b).Elem()
	Copy(ab, aa)
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			t.Errorf("(i) a[%d]=%d, b[%d]=%d", i, a[i], i, b[i])
		}
	}
	for i := len(a); i < len(b); i++ {
		if b[i] != c[i] {
			t.Errorf("(ii) b[%d]=%d, c[%d]=%d", i, b[i], i, c[i])
		} else {
			t.Logf("elem %d is okay\n", i)
		}
	}
}

func TestBigUnnamedStruct(t *testing.T) {
	b := struct{ a, b, c, d int64 }{1, 2, 3, 4}
	v := ValueOf(b)
	b1 := v.Interface().(struct {
		a, b, c, d int64
	})
	if b1.a != b.a || b1.b != b.b || b1.c != b.c || b1.d != b.d {
		t.Errorf("ValueOf(%v).Interface().(*Big) = %v", b, b1)
	}
}

type big struct {
	a, b, c, d, e int64
}

func TestBigStruct(t *testing.T) {
	b := big{1, 2, 3, 4, 5}
	v := ValueOf(b)
	b1 := v.Interface().(big)
	if b1.a != b.a || b1.b != b.b || b1.c != b.c || b1.d != b.d || b1.e != b.e {
		t.Errorf("ValueOf(%v).Interface().(big) = %v", b, b1)
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

// Simple functions for DeepEqual tests.
var (
	fn1 func()             // nil.
	fn2 func()             // nil.
	fn3 = func() { fn1() } // Not nil.
)

var deepEqualTests = []DeepEqualTest{
	// Equalities
	{nil, nil, true},
	{1, 1, true},
	{int32(1), int32(1), true},
	{0.5, 0.5, true},
	{float32(0.5), float32(0.5), true},
	{"hello", "hello", true},
	{make([]int, 10), make([]int, 10), true},
	{&[3]int{1, 2, 3}, &[3]int{1, 2, 3}, true},
	{Basic{1, 0.5}, Basic{1, 0.5}, true},
	{error(nil), error(nil), true},
	{map[int]string{1: "one", 2: "two"}, map[int]string{2: "two", 1: "one"}, true},
	{fn1, fn2, true},

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
	{fn1, fn3, false},
	{fn3, fn3, false},
	{[][]int{{1}}, [][]int{{2}}, false},

	// Nil vs empty: not the same.
	{[]int{}, []int(nil), false},
	{[]int{}, []int{}, true},
	{[]int(nil), []int(nil), true},
	{map[int]int{}, map[int]int(nil), false},
	{map[int]int{}, map[int]int{}, true},
	{map[int]int(nil), map[int]int(nil), true},

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

func TestTypeOf(t *testing.T) {
	// Special case for nil
	if typ := TypeOf(nil); typ != nil {
		t.Errorf("expected nil type for nil value; got %v", typ)
	}
	for _, test := range deepEqualTests {
		v := ValueOf(test.a)
		if !v.IsValid() {
			continue
		}
		typ := TypeOf(test.a)
		if typ != v.Type() {
			t.Errorf("TypeOf(%v) = %v, but ValueOf(%v).Type() = %v", test.a, typ, test.a, v.Type())
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

type UnexpT struct {
	m map[int]int
}

func TestDeepEqualUnexportedMap(t *testing.T) {
	// Check that DeepEqual can look at unexported fields.
	x1 := UnexpT{map[int]int{1: 2}}
	x2 := UnexpT{map[int]int{1: 2}}
	if !DeepEqual(&x1, &x2) {
		t.Error("DeepEqual(x1, x2) = false, want true")
	}

	y1 := UnexpT{map[int]int{2: 3}}
	if DeepEqual(&x1, &y1) {
		t.Error("DeepEqual(x1, y1) = true, want false")
	}
}

func check2ndField(x interface{}, offs uintptr, t *testing.T) {
	s := ValueOf(x)
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
	n := ValueOf(a).Field(0)
	if !n.IsNil() {
		t.Errorf("%v should be nil", a)
	}
}

func NotNil(a interface{}, t *testing.T) {
	n := ValueOf(a).Field(0)
	if n.IsNil() {
		t.Errorf("value of type %v should not be nil", ValueOf(a).Type().String())
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
		ty := TypeOf(ts).Field(0).Type
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
		W io.Writer
	}

	s.W = os.Stdout
	v := Indirect(ValueOf(&s)).Field(0).Interface()
	if v != s.W.(interface{}) {
		t.Error("Interface() on interface: ", v, s.W)
	}
}

func TestNilPtrValueSub(t *testing.T) {
	var pi *int
	if pv := ValueOf(pi); pv.Elem().IsValid() {
		t.Error("ValueOf((*int)(nil)).Elem().IsValid()")
	}
}

func TestMap(t *testing.T) {
	m := map[string]int{"a": 1, "b": 2}
	mv := ValueOf(m)
	if n := mv.Len(); n != len(m) {
		t.Errorf("Len = %d, want %d", n, len(m))
	}
	keys := mv.MapKeys()
	newmap := MakeMap(mv.Type())
	for k, v := range m {
		// Check that returned Keys match keys in range.
		// These aren't required to be in the same order.
		seen := false
		for _, kv := range keys {
			if kv.String() == k {
				seen = true
				break
			}
		}
		if !seen {
			t.Errorf("Missing key %q", k)
		}

		// Check that value lookup is correct.
		vv := mv.MapIndex(ValueOf(k))
		if vi := vv.Int(); vi != int64(v) {
			t.Errorf("Key %q: have value %d, want %d", k, vi, v)
		}

		// Copy into new map.
		newmap.SetMapIndex(ValueOf(k), ValueOf(v))
	}
	vv := mv.MapIndex(ValueOf("not-present"))
	if vv.IsValid() {
		t.Errorf("Invalid key: got non-nil value %s", valueToString(vv))
	}

	newm := newmap.Interface().(map[string]int)
	if len(newm) != len(m) {
		t.Errorf("length after copy: newm=%d, m=%d", len(newm), len(m))
	}

	for k, v := range newm {
		mv, ok := m[k]
		if mv != v {
			t.Errorf("newm[%q] = %d, but m[%q] = %d, %v", k, v, k, mv, ok)
		}
	}

	newmap.SetMapIndex(ValueOf("a"), Value{})
	v, ok := newm["a"]
	if ok {
		t.Errorf("newm[\"a\"] = %d after delete", v)
	}

	mv = ValueOf(&m).Elem()
	mv.Set(Zero(mv.Type()))
	if m != nil {
		t.Errorf("mv.Set(nil) failed")
	}
}

func TestNilMap(t *testing.T) {
	var m map[string]int
	mv := ValueOf(m)
	keys := mv.MapKeys()
	if len(keys) != 0 {
		t.Errorf(">0 keys for nil map: %v", keys)
	}

	// Check that value for missing key is zero.
	x := mv.MapIndex(ValueOf("hello"))
	if x.Kind() != Invalid {
		t.Errorf("m.MapIndex(\"hello\") for nil map = %v, want Invalid Value", x)
	}

	// Check big value too.
	var mbig map[string][10 << 20]byte
	x = ValueOf(mbig).MapIndex(ValueOf("hello"))
	if x.Kind() != Invalid {
		t.Errorf("mbig.MapIndex(\"hello\") for nil map = %v, want Invalid Value", x)
	}

	// Test that deletes from a nil map succeed.
	mv.SetMapIndex(ValueOf("hi"), Value{})
}

func TestChan(t *testing.T) {
	for loop := 0; loop < 2; loop++ {
		var c chan int
		var cv Value

		// check both ways to allocate channels
		switch loop {
		case 1:
			c = make(chan int, 1)
			cv = ValueOf(c)
		case 0:
			cv = MakeChan(TypeOf(c), 1)
			c = cv.Interface().(chan int)
		}

		// Send
		cv.Send(ValueOf(2))
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
		ok = cv.TrySend(ValueOf(5))
		i := <-c
		if ok {
			t.Errorf("TrySend on full chan succeeded: value %d", i)
		}

		// TrySend success
		ok = cv.TrySend(ValueOf(6))
		if !ok {
			t.Errorf("TrySend on empty chan failed")
			select {
			case x := <-c:
				t.Errorf("TrySend failed but it did send %d", x)
			default:
			}
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
	cv := MakeChan(TypeOf(c), 0)
	c = cv.Interface().(chan int)
	if cv.TrySend(ValueOf(7)) {
		t.Errorf("TrySend on sync chan succeeded")
	}
	if v, ok := cv.TryRecv(); v.IsValid() || ok {
		t.Errorf("TryRecv on sync chan succeeded: isvalid=%v ok=%v", v.IsValid(), ok)
	}

	// len/cap
	cv = MakeChan(TypeOf(c), 10)
	c = cv.Interface().(chan int)
	for i := 0; i < 3; i++ {
		c <- i
	}
	if l, m := cv.Len(), cv.Cap(); l != len(c) || m != cap(c) {
		t.Errorf("Len/Cap = %d/%d want %d/%d", l, m, len(c), cap(c))
	}
}

// caseInfo describes a single case in a select test.
type caseInfo struct {
	desc      string
	canSelect bool
	recv      Value
	closed    bool
	helper    func()
	panic     bool
}

var allselect = flag.Bool("allselect", false, "exhaustive select test")

func TestSelect(t *testing.T) {
	selectWatch.once.Do(func() { go selectWatcher() })

	var x exhaustive
	nch := 0
	newop := func(n int, cap int) (ch, val Value) {
		nch++
		if nch%101%2 == 1 {
			c := make(chan int, cap)
			ch = ValueOf(c)
			val = ValueOf(n)
		} else {
			c := make(chan string, cap)
			ch = ValueOf(c)
			val = ValueOf(fmt.Sprint(n))
		}
		return
	}

	for n := 0; x.Next(); n++ {
		if testing.Short() && n >= 1000 {
			break
		}
		if n >= 100000 && !*allselect {
			break
		}
		if n%100000 == 0 && testing.Verbose() {
			println("TestSelect", n)
		}
		var cases []SelectCase
		var info []caseInfo

		// Ready send.
		if x.Maybe() {
			ch, val := newop(len(cases), 1)
			cases = append(cases, SelectCase{
				Dir:  SelectSend,
				Chan: ch,
				Send: val,
			})
			info = append(info, caseInfo{desc: "ready send", canSelect: true})
		}

		// Ready recv.
		if x.Maybe() {
			ch, val := newop(len(cases), 1)
			ch.Send(val)
			cases = append(cases, SelectCase{
				Dir:  SelectRecv,
				Chan: ch,
			})
			info = append(info, caseInfo{desc: "ready recv", canSelect: true, recv: val})
		}

		// Blocking send.
		if x.Maybe() {
			ch, val := newop(len(cases), 0)
			cases = append(cases, SelectCase{
				Dir:  SelectSend,
				Chan: ch,
				Send: val,
			})
			// Let it execute?
			if x.Maybe() {
				f := func() { ch.Recv() }
				info = append(info, caseInfo{desc: "blocking send", helper: f})
			} else {
				info = append(info, caseInfo{desc: "blocking send"})
			}
		}

		// Blocking recv.
		if x.Maybe() {
			ch, val := newop(len(cases), 0)
			cases = append(cases, SelectCase{
				Dir:  SelectRecv,
				Chan: ch,
			})
			// Let it execute?
			if x.Maybe() {
				f := func() { ch.Send(val) }
				info = append(info, caseInfo{desc: "blocking recv", recv: val, helper: f})
			} else {
				info = append(info, caseInfo{desc: "blocking recv"})
			}
		}

		// Zero Chan send.
		if x.Maybe() {
			// Maybe include value to send.
			var val Value
			if x.Maybe() {
				val = ValueOf(100)
			}
			cases = append(cases, SelectCase{
				Dir:  SelectSend,
				Send: val,
			})
			info = append(info, caseInfo{desc: "zero Chan send"})
		}

		// Zero Chan receive.
		if x.Maybe() {
			cases = append(cases, SelectCase{
				Dir: SelectRecv,
			})
			info = append(info, caseInfo{desc: "zero Chan recv"})
		}

		// nil Chan send.
		if x.Maybe() {
			cases = append(cases, SelectCase{
				Dir:  SelectSend,
				Chan: ValueOf((chan int)(nil)),
				Send: ValueOf(101),
			})
			info = append(info, caseInfo{desc: "nil Chan send"})
		}

		// nil Chan recv.
		if x.Maybe() {
			cases = append(cases, SelectCase{
				Dir:  SelectRecv,
				Chan: ValueOf((chan int)(nil)),
			})
			info = append(info, caseInfo{desc: "nil Chan recv"})
		}

		// closed Chan send.
		if x.Maybe() {
			ch := make(chan int)
			close(ch)
			cases = append(cases, SelectCase{
				Dir:  SelectSend,
				Chan: ValueOf(ch),
				Send: ValueOf(101),
			})
			info = append(info, caseInfo{desc: "closed Chan send", canSelect: true, panic: true})
		}

		// closed Chan recv.
		if x.Maybe() {
			ch, val := newop(len(cases), 0)
			ch.Close()
			val = Zero(val.Type())
			cases = append(cases, SelectCase{
				Dir:  SelectRecv,
				Chan: ch,
			})
			info = append(info, caseInfo{desc: "closed Chan recv", canSelect: true, closed: true, recv: val})
		}

		var helper func() // goroutine to help the select complete

		// Add default? Must be last case here, but will permute.
		// Add the default if the select would otherwise
		// block forever, and maybe add it anyway.
		numCanSelect := 0
		canProceed := false
		canBlock := true
		canPanic := false
		helpers := []int{}
		for i, c := range info {
			if c.canSelect {
				canProceed = true
				canBlock = false
				numCanSelect++
				if c.panic {
					canPanic = true
				}
			} else if c.helper != nil {
				canProceed = true
				helpers = append(helpers, i)
			}
		}
		if !canProceed || x.Maybe() {
			cases = append(cases, SelectCase{
				Dir: SelectDefault,
			})
			info = append(info, caseInfo{desc: "default", canSelect: canBlock})
			numCanSelect++
		} else if canBlock {
			// Select needs to communicate with another goroutine.
			cas := &info[helpers[x.Choose(len(helpers))]]
			helper = cas.helper
			cas.canSelect = true
			numCanSelect++
		}

		// Permute cases and case info.
		// Doing too much here makes the exhaustive loop
		// too exhausting, so just do two swaps.
		for loop := 0; loop < 2; loop++ {
			i := x.Choose(len(cases))
			j := x.Choose(len(cases))
			cases[i], cases[j] = cases[j], cases[i]
			info[i], info[j] = info[j], info[i]
		}

		if helper != nil {
			// We wait before kicking off a goroutine to satisfy a blocked select.
			// The pause needs to be big enough to let the select block before
			// we run the helper, but if we lose that race once in a while it's okay: the
			// select will just proceed immediately. Not a big deal.
			// For short tests we can grow [sic] the timeout a bit without fear of taking too long
			pause := 10 * time.Microsecond
			if testing.Short() {
				pause = 100 * time.Microsecond
			}
			time.AfterFunc(pause, helper)
		}

		// Run select.
		i, recv, recvOK, panicErr := runSelect(cases, info)
		if panicErr != nil && !canPanic {
			t.Fatalf("%s\npanicked unexpectedly: %v", fmtSelect(info), panicErr)
		}
		if panicErr == nil && canPanic && numCanSelect == 1 {
			t.Fatalf("%s\nselected #%d incorrectly (should panic)", fmtSelect(info), i)
		}
		if panicErr != nil {
			continue
		}

		cas := info[i]
		if !cas.canSelect {
			recvStr := ""
			if recv.IsValid() {
				recvStr = fmt.Sprintf(", received %v, %v", recv.Interface(), recvOK)
			}
			t.Fatalf("%s\nselected #%d incorrectly%s", fmtSelect(info), i, recvStr)
			continue
		}
		if cas.panic {
			t.Fatalf("%s\nselected #%d incorrectly (case should panic)", fmtSelect(info), i)
			continue
		}

		if cases[i].Dir == SelectRecv {
			if !recv.IsValid() {
				t.Fatalf("%s\nselected #%d but got %v, %v, want %v, %v", fmtSelect(info), i, recv, recvOK, cas.recv.Interface(), !cas.closed)
			}
			if !cas.recv.IsValid() {
				t.Fatalf("%s\nselected #%d but internal error: missing recv value", fmtSelect(info), i)
			}
			if recv.Interface() != cas.recv.Interface() || recvOK != !cas.closed {
				if recv.Interface() == cas.recv.Interface() && recvOK == !cas.closed {
					t.Fatalf("%s\nselected #%d, got %#v, %v, and DeepEqual is broken on %T", fmtSelect(info), i, recv.Interface(), recvOK, recv.Interface())
				}
				t.Fatalf("%s\nselected #%d but got %#v, %v, want %#v, %v", fmtSelect(info), i, recv.Interface(), recvOK, cas.recv.Interface(), !cas.closed)
			}
		} else {
			if recv.IsValid() || recvOK {
				t.Fatalf("%s\nselected #%d but got %v, %v, want %v, %v", fmtSelect(info), i, recv, recvOK, Value{}, false)
			}
		}
	}
}

// selectWatch and the selectWatcher are a watchdog mechanism for running Select.
// If the selectWatcher notices that the select has been blocked for >1 second, it prints
// an error describing the select and panics the entire test binary.
var selectWatch struct {
	sync.Mutex
	once sync.Once
	now  time.Time
	info []caseInfo
}

func selectWatcher() {
	for {
		time.Sleep(1 * time.Second)
		selectWatch.Lock()
		if selectWatch.info != nil && time.Since(selectWatch.now) > 1*time.Second {
			fmt.Fprintf(os.Stderr, "TestSelect:\n%s blocked indefinitely\n", fmtSelect(selectWatch.info))
			panic("select stuck")
		}
		selectWatch.Unlock()
	}
}

// runSelect runs a single select test.
// It returns the values returned by Select but also returns
// a panic value if the Select panics.
func runSelect(cases []SelectCase, info []caseInfo) (chosen int, recv Value, recvOK bool, panicErr interface{}) {
	defer func() {
		panicErr = recover()

		selectWatch.Lock()
		selectWatch.info = nil
		selectWatch.Unlock()
	}()

	selectWatch.Lock()
	selectWatch.now = time.Now()
	selectWatch.info = info
	selectWatch.Unlock()

	chosen, recv, recvOK = Select(cases)
	return
}

// fmtSelect formats the information about a single select test.
func fmtSelect(info []caseInfo) string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "\nselect {\n")
	for i, cas := range info {
		fmt.Fprintf(&buf, "%d: %s", i, cas.desc)
		if cas.recv.IsValid() {
			fmt.Fprintf(&buf, " val=%#v", cas.recv.Interface())
		}
		if cas.canSelect {
			fmt.Fprintf(&buf, " canselect")
		}
		if cas.panic {
			fmt.Fprintf(&buf, " panic")
		}
		fmt.Fprintf(&buf, "\n")
	}
	fmt.Fprintf(&buf, "}")
	return buf.String()
}

type two [2]uintptr

// Difficult test for function call because of
// implicit padding between arguments.
func dummy(b byte, c int, d byte, e two, f byte, g float32, h byte) (i byte, j int, k byte, l two, m byte, n float32, o byte) {
	return b, c, d, e, f, g, h
}

func TestFunc(t *testing.T) {
	ret := ValueOf(dummy).Call([]Value{
		ValueOf(byte(10)),
		ValueOf(20),
		ValueOf(byte(30)),
		ValueOf(two{40, 50}),
		ValueOf(byte(60)),
		ValueOf(float32(70)),
		ValueOf(byte(80)),
	})
	if len(ret) != 7 {
		t.Fatalf("Call returned %d values, want 7", len(ret))
	}

	i := byte(ret[0].Uint())
	j := int(ret[1].Int())
	k := byte(ret[2].Uint())
	l := ret[3].Interface().(two)
	m := byte(ret[4].Uint())
	n := float32(ret[5].Float())
	o := byte(ret[6].Uint())

	if i != 10 || j != 20 || k != 30 || l != (two{40, 50}) || m != 60 || n != 70 || o != 80 {
		t.Errorf("Call returned %d, %d, %d, %v, %d, %g, %d; want 10, 20, 30, [40, 50], 60, 70, 80", i, j, k, l, m, n, o)
	}
}

type emptyStruct struct{}

type nonEmptyStruct struct {
	member int
}

func returnEmpty() emptyStruct {
	return emptyStruct{}
}

func takesEmpty(e emptyStruct) {
}

func returnNonEmpty(i int) nonEmptyStruct {
	return nonEmptyStruct{member: i}
}

func takesNonEmpty(n nonEmptyStruct) int {
	return n.member
}

func TestCallWithStruct(t *testing.T) {
	r := ValueOf(returnEmpty).Call(nil)
	if len(r) != 1 || r[0].Type() != TypeOf(emptyStruct{}) {
		t.Errorf("returning empty struct returned %#v instead", r)
	}
	r = ValueOf(takesEmpty).Call([]Value{ValueOf(emptyStruct{})})
	if len(r) != 0 {
		t.Errorf("takesEmpty returned values: %#v", r)
	}
	r = ValueOf(returnNonEmpty).Call([]Value{ValueOf(42)})
	if len(r) != 1 || r[0].Type() != TypeOf(nonEmptyStruct{}) || r[0].Field(0).Int() != 42 {
		t.Errorf("returnNonEmpty returned %#v", r)
	}
	r = ValueOf(takesNonEmpty).Call([]Value{ValueOf(nonEmptyStruct{member: 42})})
	if len(r) != 1 || r[0].Type() != TypeOf(1) || r[0].Int() != 42 {
		t.Errorf("takesNonEmpty returned %#v", r)
	}
}

func TestMakeFunc(t *testing.T) {
	f := dummy
	fv := MakeFunc(TypeOf(f), func(in []Value) []Value { return in })
	ValueOf(&f).Elem().Set(fv)

	// Call g with small arguments so that there is
	// something predictable (and different from the
	// correct results) in those positions on the stack.
	g := dummy
	g(1, 2, 3, two{4, 5}, 6, 7, 8)

	// Call constructed function f.
	i, j, k, l, m, n, o := f(10, 20, 30, two{40, 50}, 60, 70, 80)
	if i != 10 || j != 20 || k != 30 || l != (two{40, 50}) || m != 60 || n != 70 || o != 80 {
		t.Errorf("Call returned %d, %d, %d, %v, %d, %g, %d; want 10, 20, 30, [40, 50], 60, 70, 80", i, j, k, l, m, n, o)
	}
}

func TestMakeFuncInterface(t *testing.T) {
	fn := func(i int) int { return i }
	incr := func(in []Value) []Value {
		return []Value{ValueOf(int(in[0].Int() + 1))}
	}
	fv := MakeFunc(TypeOf(fn), incr)
	ValueOf(&fn).Elem().Set(fv)
	if r := fn(2); r != 3 {
		t.Errorf("Call returned %d, want 3", r)
	}
	if r := fv.Call([]Value{ValueOf(14)})[0].Int(); r != 15 {
		t.Errorf("Call returned %d, want 15", r)
	}
	if r := fv.Interface().(func(int) int)(26); r != 27 {
		t.Errorf("Call returned %d, want 27", r)
	}
}

func TestMakeFuncVariadic(t *testing.T) {
	// Test that variadic arguments are packed into a slice and passed as last arg
	fn := func(_ int, is ...int) []int { return nil }
	fv := MakeFunc(TypeOf(fn), func(in []Value) []Value { return in[1:2] })
	ValueOf(&fn).Elem().Set(fv)

	r := fn(1, 2, 3)
	if r[0] != 2 || r[1] != 3 {
		t.Errorf("Call returned [%v, %v]; want 2, 3", r[0], r[1])
	}

	r = fn(1, []int{2, 3}...)
	if r[0] != 2 || r[1] != 3 {
		t.Errorf("Call returned [%v, %v]; want 2, 3", r[0], r[1])
	}

	r = fv.Call([]Value{ValueOf(1), ValueOf(2), ValueOf(3)})[0].Interface().([]int)
	if r[0] != 2 || r[1] != 3 {
		t.Errorf("Call returned [%v, %v]; want 2, 3", r[0], r[1])
	}

	r = fv.CallSlice([]Value{ValueOf(1), ValueOf([]int{2, 3})})[0].Interface().([]int)
	if r[0] != 2 || r[1] != 3 {
		t.Errorf("Call returned [%v, %v]; want 2, 3", r[0], r[1])
	}

	f := fv.Interface().(func(int, ...int) []int)

	r = f(1, 2, 3)
	if r[0] != 2 || r[1] != 3 {
		t.Errorf("Call returned [%v, %v]; want 2, 3", r[0], r[1])
	}
	r = f(1, []int{2, 3}...)
	if r[0] != 2 || r[1] != 3 {
		t.Errorf("Call returned [%v, %v]; want 2, 3", r[0], r[1])
	}
}

type Point struct {
	x, y int
}

// This will be index 0.
func (p Point) AnotherMethod(scale int) int {
	return -1
}

// This will be index 1.
func (p Point) Dist(scale int) int {
	//println("Point.Dist", p.x, p.y, scale)
	return p.x*p.x*scale + p.y*p.y*scale
}

// This will be index 2.
func (p Point) GCMethod(k int) int {
	runtime.GC()
	return k + p.x
}

// This will be index 3.
func (p Point) TotalDist(points ...Point) int {
	tot := 0
	for _, q := range points {
		dx := q.x - p.x
		dy := q.y - p.y
		tot += dx*dx + dy*dy // Should call Sqrt, but it's just a test.

	}
	return tot
}

func TestMethod(t *testing.T) {
	// Non-curried method of type.
	p := Point{3, 4}
	i := TypeOf(p).Method(1).Func.Call([]Value{ValueOf(p), ValueOf(10)})[0].Int()
	if i != 250 {
		t.Errorf("Type Method returned %d; want 250", i)
	}

	m, ok := TypeOf(p).MethodByName("Dist")
	if !ok {
		t.Fatalf("method by name failed")
	}
	i = m.Func.Call([]Value{ValueOf(p), ValueOf(11)})[0].Int()
	if i != 275 {
		t.Errorf("Type MethodByName returned %d; want 275", i)
	}

	i = TypeOf(&p).Method(1).Func.Call([]Value{ValueOf(&p), ValueOf(12)})[0].Int()
	if i != 300 {
		t.Errorf("Pointer Type Method returned %d; want 300", i)
	}

	m, ok = TypeOf(&p).MethodByName("Dist")
	if !ok {
		t.Fatalf("ptr method by name failed")
	}
	i = m.Func.Call([]Value{ValueOf(&p), ValueOf(13)})[0].Int()
	if i != 325 {
		t.Errorf("Pointer Type MethodByName returned %d; want 325", i)
	}

	// Curried method of value.
	tfunc := TypeOf((func(int) int)(nil))
	v := ValueOf(p).Method(1)
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Value Method Type is %s; want %s", tt, tfunc)
	}
	i = v.Call([]Value{ValueOf(14)})[0].Int()
	if i != 350 {
		t.Errorf("Value Method returned %d; want 350", i)
	}
	v = ValueOf(p).MethodByName("Dist")
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Value MethodByName Type is %s; want %s", tt, tfunc)
	}
	i = v.Call([]Value{ValueOf(15)})[0].Int()
	if i != 375 {
		t.Errorf("Value MethodByName returned %d; want 375", i)
	}

	// Curried method of pointer.
	v = ValueOf(&p).Method(1)
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Pointer Value Method Type is %s; want %s", tt, tfunc)
	}
	i = v.Call([]Value{ValueOf(16)})[0].Int()
	if i != 400 {
		t.Errorf("Pointer Value Method returned %d; want 400", i)
	}
	v = ValueOf(&p).MethodByName("Dist")
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Pointer Value MethodByName Type is %s; want %s", tt, tfunc)
	}
	i = v.Call([]Value{ValueOf(17)})[0].Int()
	if i != 425 {
		t.Errorf("Pointer Value MethodByName returned %d; want 425", i)
	}

	// Curried method of interface value.
	// Have to wrap interface value in a struct to get at it.
	// Passing it to ValueOf directly would
	// access the underlying Point, not the interface.
	var x interface {
		Dist(int) int
	} = p
	pv := ValueOf(&x).Elem()
	v = pv.Method(0)
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Interface Method Type is %s; want %s", tt, tfunc)
	}
	i = v.Call([]Value{ValueOf(18)})[0].Int()
	if i != 450 {
		t.Errorf("Interface Method returned %d; want 450", i)
	}
	v = pv.MethodByName("Dist")
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Interface MethodByName Type is %s; want %s", tt, tfunc)
	}
	i = v.Call([]Value{ValueOf(19)})[0].Int()
	if i != 475 {
		t.Errorf("Interface MethodByName returned %d; want 475", i)
	}
}

func TestMethodValue(t *testing.T) {
	p := Point{3, 4}
	var i int64

	// Curried method of value.
	tfunc := TypeOf((func(int) int)(nil))
	v := ValueOf(p).Method(1)
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Value Method Type is %s; want %s", tt, tfunc)
	}
	i = ValueOf(v.Interface()).Call([]Value{ValueOf(10)})[0].Int()
	if i != 250 {
		t.Errorf("Value Method returned %d; want 250", i)
	}
	v = ValueOf(p).MethodByName("Dist")
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Value MethodByName Type is %s; want %s", tt, tfunc)
	}
	i = ValueOf(v.Interface()).Call([]Value{ValueOf(11)})[0].Int()
	if i != 275 {
		t.Errorf("Value MethodByName returned %d; want 275", i)
	}

	// Curried method of pointer.
	v = ValueOf(&p).Method(1)
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Pointer Value Method Type is %s; want %s", tt, tfunc)
	}
	i = ValueOf(v.Interface()).Call([]Value{ValueOf(12)})[0].Int()
	if i != 300 {
		t.Errorf("Pointer Value Method returned %d; want 300", i)
	}
	v = ValueOf(&p).MethodByName("Dist")
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Pointer Value MethodByName Type is %s; want %s", tt, tfunc)
	}
	i = ValueOf(v.Interface()).Call([]Value{ValueOf(13)})[0].Int()
	if i != 325 {
		t.Errorf("Pointer Value MethodByName returned %d; want 325", i)
	}

	// Curried method of pointer to pointer.
	pp := &p
	v = ValueOf(&pp).Elem().Method(1)
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Pointer Pointer Value Method Type is %s; want %s", tt, tfunc)
	}
	i = ValueOf(v.Interface()).Call([]Value{ValueOf(14)})[0].Int()
	if i != 350 {
		t.Errorf("Pointer Pointer Value Method returned %d; want 350", i)
	}
	v = ValueOf(&pp).Elem().MethodByName("Dist")
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Pointer Pointer Value MethodByName Type is %s; want %s", tt, tfunc)
	}
	i = ValueOf(v.Interface()).Call([]Value{ValueOf(15)})[0].Int()
	if i != 375 {
		t.Errorf("Pointer Pointer Value MethodByName returned %d; want 375", i)
	}

	// Curried method of interface value.
	// Have to wrap interface value in a struct to get at it.
	// Passing it to ValueOf directly would
	// access the underlying Point, not the interface.
	var s = struct {
		X interface {
			Dist(int) int
		}
	}{p}
	pv := ValueOf(s).Field(0)
	v = pv.Method(0)
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Interface Method Type is %s; want %s", tt, tfunc)
	}
	i = ValueOf(v.Interface()).Call([]Value{ValueOf(16)})[0].Int()
	if i != 400 {
		t.Errorf("Interface Method returned %d; want 400", i)
	}
	v = pv.MethodByName("Dist")
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Interface MethodByName Type is %s; want %s", tt, tfunc)
	}
	i = ValueOf(v.Interface()).Call([]Value{ValueOf(17)})[0].Int()
	if i != 425 {
		t.Errorf("Interface MethodByName returned %d; want 425", i)
	}
}

func TestVariadicMethodValue(t *testing.T) {
	p := Point{3, 4}
	points := []Point{{20, 21}, {22, 23}, {24, 25}}
	want := int64(p.TotalDist(points[0], points[1], points[2]))

	// Curried method of value.
	tfunc := TypeOf((func(...Point) int)(nil))
	v := ValueOf(p).Method(3)
	if tt := v.Type(); tt != tfunc {
		t.Errorf("Variadic Method Type is %s; want %s", tt, tfunc)
	}
	i := ValueOf(v.Interface()).Call([]Value{ValueOf(points[0]), ValueOf(points[1]), ValueOf(points[2])})[0].Int()
	if i != want {
		t.Errorf("Variadic Method returned %d; want %d", i, want)
	}
	i = ValueOf(v.Interface()).CallSlice([]Value{ValueOf(points)})[0].Int()
	if i != want {
		t.Errorf("Variadic Method CallSlice returned %d; want %d", i, want)
	}

	f := v.Interface().(func(...Point) int)
	i = int64(f(points[0], points[1], points[2]))
	if i != want {
		t.Errorf("Variadic Method Interface returned %d; want %d", i, want)
	}
	i = int64(f(points...))
	if i != want {
		t.Errorf("Variadic Method Interface Slice returned %d; want %d", i, want)
	}
}

// Reflect version of $GOROOT/test/method5.go

// Concrete types implementing M method.
// Smaller than a word, word-sized, larger than a word.
// Value and pointer receivers.

type Tinter interface {
	M(int, byte) (byte, int)
}

type Tsmallv byte

func (v Tsmallv) M(x int, b byte) (byte, int) { return b, x + int(v) }

type Tsmallp byte

func (p *Tsmallp) M(x int, b byte) (byte, int) { return b, x + int(*p) }

type Twordv uintptr

func (v Twordv) M(x int, b byte) (byte, int) { return b, x + int(v) }

type Twordp uintptr

func (p *Twordp) M(x int, b byte) (byte, int) { return b, x + int(*p) }

type Tbigv [2]uintptr

func (v Tbigv) M(x int, b byte) (byte, int) { return b, x + int(v[0]) + int(v[1]) }

type Tbigp [2]uintptr

func (p *Tbigp) M(x int, b byte) (byte, int) { return b, x + int(p[0]) + int(p[1]) }

// Again, with an unexported method.

type tsmallv byte

func (v tsmallv) m(x int, b byte) (byte, int) { return b, x + int(v) }

type tsmallp byte

func (p *tsmallp) m(x int, b byte) (byte, int) { return b, x + int(*p) }

type twordv uintptr

func (v twordv) m(x int, b byte) (byte, int) { return b, x + int(v) }

type twordp uintptr

func (p *twordp) m(x int, b byte) (byte, int) { return b, x + int(*p) }

type tbigv [2]uintptr

func (v tbigv) m(x int, b byte) (byte, int) { return b, x + int(v[0]) + int(v[1]) }

type tbigp [2]uintptr

func (p *tbigp) m(x int, b byte) (byte, int) { return b, x + int(p[0]) + int(p[1]) }

type tinter interface {
	m(int, byte) (byte, int)
}

// Embedding via pointer.

type Tm1 struct {
	Tm2
}

type Tm2 struct {
	*Tm3
}

type Tm3 struct {
	*Tm4
}

type Tm4 struct {
}

func (t4 Tm4) M(x int, b byte) (byte, int) { return b, x + 40 }

func TestMethod5(t *testing.T) {
	CheckF := func(name string, f func(int, byte) (byte, int), inc int) {
		b, x := f(1000, 99)
		if b != 99 || x != 1000+inc {
			t.Errorf("%s(1000, 99) = %v, %v, want 99, %v", name, b, x, 1000+inc)
		}
	}

	CheckV := func(name string, i Value, inc int) {
		bx := i.Method(0).Call([]Value{ValueOf(1000), ValueOf(byte(99))})
		b := bx[0].Interface()
		x := bx[1].Interface()
		if b != byte(99) || x != 1000+inc {
			t.Errorf("direct %s.M(1000, 99) = %v, %v, want 99, %v", name, b, x, 1000+inc)
		}

		CheckF(name+".M", i.Method(0).Interface().(func(int, byte) (byte, int)), inc)
	}

	var TinterType = TypeOf(new(Tinter)).Elem()
	var tinterType = TypeOf(new(tinter)).Elem()

	CheckI := func(name string, i interface{}, inc int) {
		v := ValueOf(i)
		CheckV(name, v, inc)
		CheckV("(i="+name+")", v.Convert(TinterType), inc)
	}

	sv := Tsmallv(1)
	CheckI("sv", sv, 1)
	CheckI("&sv", &sv, 1)

	sp := Tsmallp(2)
	CheckI("&sp", &sp, 2)

	wv := Twordv(3)
	CheckI("wv", wv, 3)
	CheckI("&wv", &wv, 3)

	wp := Twordp(4)
	CheckI("&wp", &wp, 4)

	bv := Tbigv([2]uintptr{5, 6})
	CheckI("bv", bv, 11)
	CheckI("&bv", &bv, 11)

	bp := Tbigp([2]uintptr{7, 8})
	CheckI("&bp", &bp, 15)

	t4 := Tm4{}
	t3 := Tm3{&t4}
	t2 := Tm2{&t3}
	t1 := Tm1{t2}
	CheckI("t4", t4, 40)
	CheckI("&t4", &t4, 40)
	CheckI("t3", t3, 40)
	CheckI("&t3", &t3, 40)
	CheckI("t2", t2, 40)
	CheckI("&t2", &t2, 40)
	CheckI("t1", t1, 40)
	CheckI("&t1", &t1, 40)

	methodShouldPanic := func(name string, i interface{}) {
		v := ValueOf(i)
		m := v.Method(0)
		shouldPanic(func() { m.Call([]Value{ValueOf(1000), ValueOf(byte(99))}) })
		shouldPanic(func() { m.Interface() })

		v = v.Convert(tinterType)
		m = v.Method(0)
		shouldPanic(func() { m.Call([]Value{ValueOf(1000), ValueOf(byte(99))}) })
		shouldPanic(func() { m.Interface() })
	}

	_sv := tsmallv(1)
	methodShouldPanic("_sv", _sv)
	methodShouldPanic("&_sv", &_sv)

	_sp := tsmallp(2)
	methodShouldPanic("&_sp", &_sp)

	_wv := twordv(3)
	methodShouldPanic("_wv", _wv)
	methodShouldPanic("&_wv", &_wv)

	_wp := twordp(4)
	methodShouldPanic("&_wp", &_wp)

	_bv := tbigv([2]uintptr{5, 6})
	methodShouldPanic("_bv", _bv)
	methodShouldPanic("&_bv", &_bv)

	_bp := tbigp([2]uintptr{7, 8})
	methodShouldPanic("&_bp", &_bp)

	var tnil Tinter
	vnil := ValueOf(&tnil).Elem()
	shouldPanic(func() { vnil.Method(0) })
}

func TestInterfaceSet(t *testing.T) {
	p := &Point{3, 4}

	var s struct {
		I interface{}
		P interface {
			Dist(int) int
		}
	}
	sv := ValueOf(&s).Elem()
	sv.Field(0).Set(ValueOf(p))
	if q := s.I.(*Point); q != p {
		t.Errorf("i: have %p want %p", q, p)
	}

	pv := sv.Field(1)
	pv.Set(ValueOf(p))
	if q := s.P.(*Point); q != p {
		t.Errorf("i: have %p want %p", q, p)
	}

	i := pv.Method(0).Call([]Value{ValueOf(10)})[0].Int()
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
	type1 := TypeOf(t1)
	if field, ok = type1.FieldByName("int"); !ok {
		t.Fatal("no field 'int'")
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
	A, B, C int
	D1
	D2
}

type S1 struct {
	B int
	S0
}

type S2 struct {
	A int
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
	D, E int
	*S1y
}

type S4 struct {
	*S4
	A int
}

// The X in S6 and S7 annihilate, but they also block the X in S8.S9.
type S5 struct {
	S6
	S7
	S8
}

type S6 struct {
	X int
}

type S7 S6

type S8 struct {
	S9
}

type S9 struct {
	X int
	Y int
}

// The X in S11.S6 and S12.S6 annihilate, but they also block the X in S13.S8.S9.
type S10 struct {
	S11
	S12
	S13
}

type S11 struct {
	S6
}

type S12 struct {
	S6
}

type S13 struct {
	S8
}

// The X in S15.S11.S1 and S16.S11.S1 annihilate.
type S14 struct {
	S15
	S16
}

type S15 struct {
	S11
}

type S16 struct {
	S11
}

var fieldTests = []FTest{
	{struct{}{}, "", nil, 0},
	{struct{}{}, "Foo", nil, 0},
	{S0{A: 'a'}, "A", []int{0}, 'a'},
	{S0{}, "D", nil, 0},
	{S1{S0: S0{A: 'a'}}, "A", []int{1, 0}, 'a'},
	{S1{B: 'b'}, "B", []int{0}, 'b'},
	{S1{}, "S0", []int{1}, 0},
	{S1{S0: S0{C: 'c'}}, "C", []int{1, 2}, 'c'},
	{S2{A: 'a'}, "A", []int{0}, 'a'},
	{S2{}, "S1", []int{1}, 0},
	{S2{S1: &S1{B: 'b'}}, "B", []int{1, 0}, 'b'},
	{S2{S1: &S1{S0: S0{C: 'c'}}}, "C", []int{1, 1, 2}, 'c'},
	{S2{}, "D", nil, 0},
	{S3{}, "S1", nil, 0},
	{S3{S2: S2{A: 'a'}}, "A", []int{1, 0}, 'a'},
	{S3{}, "B", nil, 0},
	{S3{D: 'd'}, "D", []int{2}, 0},
	{S3{E: 'e'}, "E", []int{3}, 'e'},
	{S4{A: 'a'}, "A", []int{1}, 'a'},
	{S4{}, "B", nil, 0},
	{S5{}, "X", nil, 0},
	{S5{}, "Y", []int{2, 0, 1}, 0},
	{S10{}, "X", nil, 0},
	{S10{}, "Y", []int{2, 0, 0, 1}, 0},
	{S14{}, "X", nil, 0},
}

func TestFieldByIndex(t *testing.T) {
	for _, test := range fieldTests {
		s := TypeOf(test.s)
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
			v := ValueOf(test.s).FieldByIndex(test.index)
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
		s := TypeOf(test.s)
		f, found := s.FieldByName(test.name)
		if found {
			if test.index != nil {
				// Verify field depth and index.
				if len(f.Index) != len(test.index) {
					t.Errorf("%s.%s depth %d; want %d: %v vs %v", s.Name(), test.name, len(f.Index), len(test.index), f.Index, test.index)
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
			v := ValueOf(test.s).FieldByName(test.name)
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
	tests := []struct {
		t    Type
		path string
	}{
		{TypeOf(&base64.Encoding{}).Elem(), "encoding/base64"},
		{TypeOf(int(0)), ""},
		{TypeOf(int8(0)), ""},
		{TypeOf(int16(0)), ""},
		{TypeOf(int32(0)), ""},
		{TypeOf(int64(0)), ""},
		{TypeOf(uint(0)), ""},
		{TypeOf(uint8(0)), ""},
		{TypeOf(uint16(0)), ""},
		{TypeOf(uint32(0)), ""},
		{TypeOf(uint64(0)), ""},
		{TypeOf(uintptr(0)), ""},
		{TypeOf(float32(0)), ""},
		{TypeOf(float64(0)), ""},
		{TypeOf(complex64(0)), ""},
		{TypeOf(complex128(0)), ""},
		{TypeOf(byte(0)), ""},
		{TypeOf(rune(0)), ""},
		{TypeOf([]byte(nil)), ""},
		{TypeOf([]rune(nil)), ""},
		{TypeOf(string("")), ""},
		{TypeOf((*interface{})(nil)).Elem(), ""},
		{TypeOf((*byte)(nil)), ""},
		{TypeOf((*rune)(nil)), ""},
		{TypeOf((*int64)(nil)), ""},
		{TypeOf(map[string]int{}), ""},
		{TypeOf((*error)(nil)).Elem(), ""},
	}
	for _, test := range tests {
		if path := test.t.PkgPath(); path != test.path {
			t.Errorf("%v.PkgPath() = %q, want %q", test.t, path, test.path)
		}
	}
}

func TestVariadicType(t *testing.T) {
	// Test example from Type documentation.
	var f func(x int, y ...float64)
	typ := TypeOf(f)
	if typ.NumIn() == 2 && typ.In(0) == TypeOf(int(0)) {
		sl := typ.In(1)
		if sl.Kind() == Slice {
			if sl.Elem() == TypeOf(0.0) {
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
	typ := TypeOf((*outer)(nil))
	if typ.NumMethod() != 1 || typ.Method(0).Func.Pointer() != ValueOf((*outer).m).Pointer() {
		t.Errorf("Wrong method table for outer: (m=%p)", (*outer).m)
		for i := 0; i < typ.NumMethod(); i++ {
			m := typ.Method(i)
			t.Errorf("\t%d: %s %#x\n", i, m.Name, m.Func.Pointer())
		}
	}
}

type InnerInt struct {
	X int
}

type OuterInt struct {
	Y int
	InnerInt
}

func (i *InnerInt) M() int {
	return i.X
}

func TestEmbeddedMethods(t *testing.T) {
	typ := TypeOf((*OuterInt)(nil))
	if typ.NumMethod() != 1 || typ.Method(0).Func.Pointer() != ValueOf((*OuterInt).M).Pointer() {
		t.Errorf("Wrong method table for OuterInt: (m=%p)", (*OuterInt).M)
		for i := 0; i < typ.NumMethod(); i++ {
			m := typ.Method(i)
			t.Errorf("\t%d: %s %#x\n", i, m.Name, m.Func.Pointer())
		}
	}

	i := &InnerInt{3}
	if v := ValueOf(i).Method(0).Call(nil)[0].Int(); v != 3 {
		t.Errorf("i.M() = %d, want 3", v)
	}

	o := &OuterInt{1, InnerInt{2}}
	if v := ValueOf(o).Method(0).Call(nil)[0].Int(); v != 2 {
		t.Errorf("i.M() = %d, want 2", v)
	}

	f := (*OuterInt).M
	if v := f(o); v != 2 {
		t.Errorf("f(o) = %d, want 2", v)
	}
}

func TestPtrTo(t *testing.T) {
	var i int

	typ := TypeOf(i)
	for i = 0; i < 100; i++ {
		typ = PtrTo(typ)
	}
	for i = 0; i < 100; i++ {
		typ = typ.Elem()
	}
	if typ != TypeOf(i) {
		t.Errorf("after 100 PtrTo and Elem, have %s, want %s", typ, TypeOf(i))
	}
}

func TestPtrToGC(t *testing.T) {
	type T *uintptr
	tt := TypeOf(T(nil))
	pt := PtrTo(tt)
	const n = 100
	var x []interface{}
	for i := 0; i < n; i++ {
		v := New(pt)
		p := new(*uintptr)
		*p = new(uintptr)
		**p = uintptr(i)
		v.Elem().Set(ValueOf(p).Convert(pt))
		x = append(x, v.Interface())
	}
	runtime.GC()

	for i, xi := range x {
		k := ValueOf(xi).Elem().Elem().Elem().Interface().(uintptr)
		if k != uintptr(i) {
			t.Errorf("lost x[%d] = %d, want %d", i, k, i)
		}
	}
}

func TestAddr(t *testing.T) {
	var p struct {
		X, Y int
	}

	v := ValueOf(&p)
	v = v.Elem()
	v = v.Addr()
	v = v.Elem()
	v = v.Field(0)
	v.SetInt(2)
	if p.X != 2 {
		t.Errorf("Addr.Elem.Set failed to set value")
	}

	// Again but take address of the ValueOf value.
	// Exercises generation of PtrTypes not present in the binary.
	q := &p
	v = ValueOf(&q).Elem()
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
	v = ValueOf(&qq).Elem()
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

	// Verify that taking the address of a type gives us a pointer
	// which we can convert back using the usual interface
	// notation.
	var s struct {
		B *bool
	}
	ps := ValueOf(&s).Elem().Field(0).Addr().Interface()
	*(ps.(**bool)) = new(bool)
	if s.B == nil {
		t.Errorf("Addr.Interface direct assignment failed")
	}
}

func noAlloc(t *testing.T, n int, f func(int)) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	if runtime.GOMAXPROCS(0) > 1 {
		t.Skip("skipping; GOMAXPROCS>1")
	}
	i := -1
	allocs := testing.AllocsPerRun(n, func() {
		f(i)
		i++
	})
	if allocs > 0 {
		t.Errorf("%d iterations: got %v mallocs, want 0", n, allocs)
	}
}

func TestAllocations(t *testing.T) {
	noAlloc(t, 100, func(j int) {
		var i interface{}
		var v Value

		// We can uncomment this when compiler escape analysis
		// is good enough to see that the integer assigned to i
		// does not escape and therefore need not be allocated.
		//
		// i = 42 + j
		// v = ValueOf(i)
		// if int(v.Int()) != 42+j {
		// 	panic("wrong int")
		// }

		i = func(j int) int { return j }
		v = ValueOf(i)
		if v.Interface().(func(int) int)(j) != j {
			panic("wrong result")
		}
	})
}

func TestSmallNegativeInt(t *testing.T) {
	i := int16(-1)
	v := ValueOf(i)
	if v.Int() != -1 {
		t.Errorf("int16(-1).Int() returned %v", v.Int())
	}
}

func TestIndex(t *testing.T) {
	xs := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	v := ValueOf(xs).Index(3).Interface().(byte)
	if v != xs[3] {
		t.Errorf("xs.Index(3) = %v; expected %v", v, xs[3])
	}
	xa := [8]byte{10, 20, 30, 40, 50, 60, 70, 80}
	v = ValueOf(xa).Index(2).Interface().(byte)
	if v != xa[2] {
		t.Errorf("xa.Index(2) = %v; expected %v", v, xa[2])
	}
	s := "0123456789"
	v = ValueOf(s).Index(3).Interface().(byte)
	if v != s[3] {
		t.Errorf("s.Index(3) = %v; expected %v", v, s[3])
	}
}

func TestSlice(t *testing.T) {
	xs := []int{1, 2, 3, 4, 5, 6, 7, 8}
	v := ValueOf(xs).Slice(3, 5).Interface().([]int)
	if len(v) != 2 {
		t.Errorf("len(xs.Slice(3, 5)) = %d", len(v))
	}
	if cap(v) != 5 {
		t.Errorf("cap(xs.Slice(3, 5)) = %d", cap(v))
	}
	if !DeepEqual(v[0:5], xs[3:]) {
		t.Errorf("xs.Slice(3, 5)[0:5] = %v", v[0:5])
	}
	xa := [8]int{10, 20, 30, 40, 50, 60, 70, 80}
	v = ValueOf(&xa).Elem().Slice(2, 5).Interface().([]int)
	if len(v) != 3 {
		t.Errorf("len(xa.Slice(2, 5)) = %d", len(v))
	}
	if cap(v) != 6 {
		t.Errorf("cap(xa.Slice(2, 5)) = %d", cap(v))
	}
	if !DeepEqual(v[0:6], xa[2:]) {
		t.Errorf("xs.Slice(2, 5)[0:6] = %v", v[0:6])
	}
	s := "0123456789"
	vs := ValueOf(s).Slice(3, 5).Interface().(string)
	if vs != s[3:5] {
		t.Errorf("s.Slice(3, 5) = %q; expected %q", vs, s[3:5])
	}

	rv := ValueOf(&xs).Elem()
	rv = rv.Slice(3, 4)
	ptr2 := rv.Pointer()
	rv = rv.Slice(5, 5)
	ptr3 := rv.Pointer()
	if ptr3 != ptr2 {
		t.Errorf("xs.Slice(3,4).Slice3(5,5).Pointer() = %#x, want %#x", ptr3, ptr2)
	}
}

func TestSlice3(t *testing.T) {
	xs := []int{1, 2, 3, 4, 5, 6, 7, 8}
	v := ValueOf(xs).Slice3(3, 5, 7).Interface().([]int)
	if len(v) != 2 {
		t.Errorf("len(xs.Slice3(3, 5, 7)) = %d", len(v))
	}
	if cap(v) != 4 {
		t.Errorf("cap(xs.Slice3(3, 5, 7)) = %d", cap(v))
	}
	if !DeepEqual(v[0:4], xs[3:7:7]) {
		t.Errorf("xs.Slice3(3, 5, 7)[0:4] = %v", v[0:4])
	}
	rv := ValueOf(&xs).Elem()
	shouldPanic(func() { rv.Slice3(1, 2, 1) })
	shouldPanic(func() { rv.Slice3(1, 1, 11) })
	shouldPanic(func() { rv.Slice3(2, 2, 1) })

	xa := [8]int{10, 20, 30, 40, 50, 60, 70, 80}
	v = ValueOf(&xa).Elem().Slice3(2, 5, 6).Interface().([]int)
	if len(v) != 3 {
		t.Errorf("len(xa.Slice(2, 5, 6)) = %d", len(v))
	}
	if cap(v) != 4 {
		t.Errorf("cap(xa.Slice(2, 5, 6)) = %d", cap(v))
	}
	if !DeepEqual(v[0:4], xa[2:6:6]) {
		t.Errorf("xs.Slice(2, 5, 6)[0:4] = %v", v[0:4])
	}
	rv = ValueOf(&xa).Elem()
	shouldPanic(func() { rv.Slice3(1, 2, 1) })
	shouldPanic(func() { rv.Slice3(1, 1, 11) })
	shouldPanic(func() { rv.Slice3(2, 2, 1) })

	s := "hello world"
	rv = ValueOf(&s).Elem()
	shouldPanic(func() { rv.Slice3(1, 2, 3) })

	rv = ValueOf(&xs).Elem()
	rv = rv.Slice3(3, 5, 7)
	ptr2 := rv.Pointer()
	rv = rv.Slice3(4, 4, 4)
	ptr3 := rv.Pointer()
	if ptr3 != ptr2 {
		t.Errorf("xs.Slice3(3,5,7).Slice3(4,4,4).Pointer() = %#x, want %#x", ptr3, ptr2)
	}
}

func TestSetLenCap(t *testing.T) {
	xs := []int{1, 2, 3, 4, 5, 6, 7, 8}
	xa := [8]int{10, 20, 30, 40, 50, 60, 70, 80}

	vs := ValueOf(&xs).Elem()
	shouldPanic(func() { vs.SetLen(10) })
	shouldPanic(func() { vs.SetCap(10) })
	shouldPanic(func() { vs.SetLen(-1) })
	shouldPanic(func() { vs.SetCap(-1) })
	shouldPanic(func() { vs.SetCap(6) }) // smaller than len
	vs.SetLen(5)
	if len(xs) != 5 || cap(xs) != 8 {
		t.Errorf("after SetLen(5), len, cap = %d, %d, want 5, 8", len(xs), cap(xs))
	}
	vs.SetCap(6)
	if len(xs) != 5 || cap(xs) != 6 {
		t.Errorf("after SetCap(6), len, cap = %d, %d, want 5, 6", len(xs), cap(xs))
	}
	vs.SetCap(5)
	if len(xs) != 5 || cap(xs) != 5 {
		t.Errorf("after SetCap(5), len, cap = %d, %d, want 5, 5", len(xs), cap(xs))
	}
	shouldPanic(func() { vs.SetCap(4) }) // smaller than len
	shouldPanic(func() { vs.SetLen(6) }) // bigger than cap

	va := ValueOf(&xa).Elem()
	shouldPanic(func() { va.SetLen(8) })
	shouldPanic(func() { va.SetCap(8) })
}

func TestVariadic(t *testing.T) {
	var b bytes.Buffer
	V := ValueOf

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

func TestFuncArg(t *testing.T) {
	f1 := func(i int, f func(int) int) int { return f(i) }
	f2 := func(i int) int { return i + 1 }
	r := ValueOf(f1).Call([]Value{ValueOf(100), ValueOf(f2)})
	if r[0].Int() != 101 {
		t.Errorf("function returned %d, want 101", r[0].Int())
	}
}

func TestStructArg(t *testing.T) {
	type padded struct {
		B string
		C int32
	}
	var (
		gotA  padded
		gotB  uint32
		wantA = padded{"3", 4}
		wantB = uint32(5)
	)
	f := func(a padded, b uint32) {
		gotA, gotB = a, b
	}
	ValueOf(f).Call([]Value{ValueOf(wantA), ValueOf(wantB)})
	if gotA != wantA || gotB != wantB {
		t.Errorf("function called with (%v, %v), want (%v, %v)", gotA, gotB, wantA, wantB)
	}
}

var tagGetTests = []struct {
	Tag   StructTag
	Key   string
	Value string
}{
	{`protobuf:"PB(1,2)"`, `protobuf`, `PB(1,2)`},
	{`protobuf:"PB(1,2)"`, `foo`, ``},
	{`protobuf:"PB(1,2)"`, `rotobuf`, ``},
	{`protobuf:"PB(1,2)" json:"name"`, `json`, `name`},
	{`protobuf:"PB(1,2)" json:"name"`, `protobuf`, `PB(1,2)`},
}

func TestTagGet(t *testing.T) {
	for _, tt := range tagGetTests {
		if v := tt.Tag.Get(tt.Key); v != tt.Value {
			t.Errorf("StructTag(%#q).Get(%#q) = %#q, want %#q", tt.Tag, tt.Key, v, tt.Value)
		}
	}
}

func TestBytes(t *testing.T) {
	type B []byte
	x := B{1, 2, 3, 4}
	y := ValueOf(x).Bytes()
	if !bytes.Equal(x, y) {
		t.Fatalf("ValueOf(%v).Bytes() = %v", x, y)
	}
	if &x[0] != &y[0] {
		t.Errorf("ValueOf(%p).Bytes() = %p", &x[0], &y[0])
	}
}

func TestSetBytes(t *testing.T) {
	type B []byte
	var x B
	y := []byte{1, 2, 3, 4}
	ValueOf(&x).Elem().SetBytes(y)
	if !bytes.Equal(x, y) {
		t.Fatalf("ValueOf(%v).Bytes() = %v", x, y)
	}
	if &x[0] != &y[0] {
		t.Errorf("ValueOf(%p).Bytes() = %p", &x[0], &y[0])
	}
}

type Private struct {
	x int
	y **int
}

func (p *Private) m() {
}

type Public struct {
	X int
	Y **int
}

func (p *Public) M() {
}

func TestUnexported(t *testing.T) {
	var pub Public
	v := ValueOf(&pub)
	isValid(v.Elem().Field(0))
	isValid(v.Elem().Field(1))
	isValid(v.Elem().FieldByName("X"))
	isValid(v.Elem().FieldByName("Y"))
	isValid(v.Type().Method(0).Func)
	isNonNil(v.Elem().Field(0).Interface())
	isNonNil(v.Elem().Field(1).Interface())
	isNonNil(v.Elem().FieldByName("X").Interface())
	isNonNil(v.Elem().FieldByName("Y").Interface())
	isNonNil(v.Type().Method(0).Func.Interface())

	var priv Private
	v = ValueOf(&priv)
	isValid(v.Elem().Field(0))
	isValid(v.Elem().Field(1))
	isValid(v.Elem().FieldByName("x"))
	isValid(v.Elem().FieldByName("y"))
	isValid(v.Type().Method(0).Func)
	shouldPanic(func() { v.Elem().Field(0).Interface() })
	shouldPanic(func() { v.Elem().Field(1).Interface() })
	shouldPanic(func() { v.Elem().FieldByName("x").Interface() })
	shouldPanic(func() { v.Elem().FieldByName("y").Interface() })
	shouldPanic(func() { v.Type().Method(0).Func.Interface() })
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("did not panic")
		}
	}()
	f()
}

func isNonNil(x interface{}) {
	if x == nil {
		panic("nil interface")
	}
}

func isValid(v Value) {
	if !v.IsValid() {
		panic("zero Value")
	}
}

func TestAlias(t *testing.T) {
	x := string("hello")
	v := ValueOf(&x).Elem()
	oldvalue := v.Interface()
	v.SetString("world")
	newvalue := v.Interface()

	if oldvalue != "hello" || newvalue != "world" {
		t.Errorf("aliasing: old=%q new=%q, want hello, world", oldvalue, newvalue)
	}
}

var V = ValueOf

func EmptyInterfaceV(x interface{}) Value {
	return ValueOf(&x).Elem()
}

func ReaderV(x io.Reader) Value {
	return ValueOf(&x).Elem()
}

func ReadWriterV(x io.ReadWriter) Value {
	return ValueOf(&x).Elem()
}

type Empty struct{}
type MyString string
type MyBytes []byte
type MyRunes []int32
type MyFunc func()
type MyByte byte

var convertTests = []struct {
	in  Value
	out Value
}{
	// numbers
	/*
		Edit .+1,/\*\//-1>cat >/tmp/x.go && go run /tmp/x.go

		package main

		import "fmt"

		var numbers = []string{
			"int8", "uint8", "int16", "uint16",
			"int32", "uint32", "int64", "uint64",
			"int", "uint", "uintptr",
			"float32", "float64",
		}

		func main() {
			// all pairs but in an unusual order,
			// to emit all the int8, uint8 cases
			// before n grows too big.
			n := 1
			for i, f := range numbers {
				for _, g := range numbers[i:] {
					fmt.Printf("\t{V(%s(%d)), V(%s(%d))},\n", f, n, g, n)
					n++
					if f != g {
						fmt.Printf("\t{V(%s(%d)), V(%s(%d))},\n", g, n, f, n)
						n++
					}
				}
			}
		}
	*/
	{V(int8(1)), V(int8(1))},
	{V(int8(2)), V(uint8(2))},
	{V(uint8(3)), V(int8(3))},
	{V(int8(4)), V(int16(4))},
	{V(int16(5)), V(int8(5))},
	{V(int8(6)), V(uint16(6))},
	{V(uint16(7)), V(int8(7))},
	{V(int8(8)), V(int32(8))},
	{V(int32(9)), V(int8(9))},
	{V(int8(10)), V(uint32(10))},
	{V(uint32(11)), V(int8(11))},
	{V(int8(12)), V(int64(12))},
	{V(int64(13)), V(int8(13))},
	{V(int8(14)), V(uint64(14))},
	{V(uint64(15)), V(int8(15))},
	{V(int8(16)), V(int(16))},
	{V(int(17)), V(int8(17))},
	{V(int8(18)), V(uint(18))},
	{V(uint(19)), V(int8(19))},
	{V(int8(20)), V(uintptr(20))},
	{V(uintptr(21)), V(int8(21))},
	{V(int8(22)), V(float32(22))},
	{V(float32(23)), V(int8(23))},
	{V(int8(24)), V(float64(24))},
	{V(float64(25)), V(int8(25))},
	{V(uint8(26)), V(uint8(26))},
	{V(uint8(27)), V(int16(27))},
	{V(int16(28)), V(uint8(28))},
	{V(uint8(29)), V(uint16(29))},
	{V(uint16(30)), V(uint8(30))},
	{V(uint8(31)), V(int32(31))},
	{V(int32(32)), V(uint8(32))},
	{V(uint8(33)), V(uint32(33))},
	{V(uint32(34)), V(uint8(34))},
	{V(uint8(35)), V(int64(35))},
	{V(int64(36)), V(uint8(36))},
	{V(uint8(37)), V(uint64(37))},
	{V(uint64(38)), V(uint8(38))},
	{V(uint8(39)), V(int(39))},
	{V(int(40)), V(uint8(40))},
	{V(uint8(41)), V(uint(41))},
	{V(uint(42)), V(uint8(42))},
	{V(uint8(43)), V(uintptr(43))},
	{V(uintptr(44)), V(uint8(44))},
	{V(uint8(45)), V(float32(45))},
	{V(float32(46)), V(uint8(46))},
	{V(uint8(47)), V(float64(47))},
	{V(float64(48)), V(uint8(48))},
	{V(int16(49)), V(int16(49))},
	{V(int16(50)), V(uint16(50))},
	{V(uint16(51)), V(int16(51))},
	{V(int16(52)), V(int32(52))},
	{V(int32(53)), V(int16(53))},
	{V(int16(54)), V(uint32(54))},
	{V(uint32(55)), V(int16(55))},
	{V(int16(56)), V(int64(56))},
	{V(int64(57)), V(int16(57))},
	{V(int16(58)), V(uint64(58))},
	{V(uint64(59)), V(int16(59))},
	{V(int16(60)), V(int(60))},
	{V(int(61)), V(int16(61))},
	{V(int16(62)), V(uint(62))},
	{V(uint(63)), V(int16(63))},
	{V(int16(64)), V(uintptr(64))},
	{V(uintptr(65)), V(int16(65))},
	{V(int16(66)), V(float32(66))},
	{V(float32(67)), V(int16(67))},
	{V(int16(68)), V(float64(68))},
	{V(float64(69)), V(int16(69))},
	{V(uint16(70)), V(uint16(70))},
	{V(uint16(71)), V(int32(71))},
	{V(int32(72)), V(uint16(72))},
	{V(uint16(73)), V(uint32(73))},
	{V(uint32(74)), V(uint16(74))},
	{V(uint16(75)), V(int64(75))},
	{V(int64(76)), V(uint16(76))},
	{V(uint16(77)), V(uint64(77))},
	{V(uint64(78)), V(uint16(78))},
	{V(uint16(79)), V(int(79))},
	{V(int(80)), V(uint16(80))},
	{V(uint16(81)), V(uint(81))},
	{V(uint(82)), V(uint16(82))},
	{V(uint16(83)), V(uintptr(83))},
	{V(uintptr(84)), V(uint16(84))},
	{V(uint16(85)), V(float32(85))},
	{V(float32(86)), V(uint16(86))},
	{V(uint16(87)), V(float64(87))},
	{V(float64(88)), V(uint16(88))},
	{V(int32(89)), V(int32(89))},
	{V(int32(90)), V(uint32(90))},
	{V(uint32(91)), V(int32(91))},
	{V(int32(92)), V(int64(92))},
	{V(int64(93)), V(int32(93))},
	{V(int32(94)), V(uint64(94))},
	{V(uint64(95)), V(int32(95))},
	{V(int32(96)), V(int(96))},
	{V(int(97)), V(int32(97))},
	{V(int32(98)), V(uint(98))},
	{V(uint(99)), V(int32(99))},
	{V(int32(100)), V(uintptr(100))},
	{V(uintptr(101)), V(int32(101))},
	{V(int32(102)), V(float32(102))},
	{V(float32(103)), V(int32(103))},
	{V(int32(104)), V(float64(104))},
	{V(float64(105)), V(int32(105))},
	{V(uint32(106)), V(uint32(106))},
	{V(uint32(107)), V(int64(107))},
	{V(int64(108)), V(uint32(108))},
	{V(uint32(109)), V(uint64(109))},
	{V(uint64(110)), V(uint32(110))},
	{V(uint32(111)), V(int(111))},
	{V(int(112)), V(uint32(112))},
	{V(uint32(113)), V(uint(113))},
	{V(uint(114)), V(uint32(114))},
	{V(uint32(115)), V(uintptr(115))},
	{V(uintptr(116)), V(uint32(116))},
	{V(uint32(117)), V(float32(117))},
	{V(float32(118)), V(uint32(118))},
	{V(uint32(119)), V(float64(119))},
	{V(float64(120)), V(uint32(120))},
	{V(int64(121)), V(int64(121))},
	{V(int64(122)), V(uint64(122))},
	{V(uint64(123)), V(int64(123))},
	{V(int64(124)), V(int(124))},
	{V(int(125)), V(int64(125))},
	{V(int64(126)), V(uint(126))},
	{V(uint(127)), V(int64(127))},
	{V(int64(128)), V(uintptr(128))},
	{V(uintptr(129)), V(int64(129))},
	{V(int64(130)), V(float32(130))},
	{V(float32(131)), V(int64(131))},
	{V(int64(132)), V(float64(132))},
	{V(float64(133)), V(int64(133))},
	{V(uint64(134)), V(uint64(134))},
	{V(uint64(135)), V(int(135))},
	{V(int(136)), V(uint64(136))},
	{V(uint64(137)), V(uint(137))},
	{V(uint(138)), V(uint64(138))},
	{V(uint64(139)), V(uintptr(139))},
	{V(uintptr(140)), V(uint64(140))},
	{V(uint64(141)), V(float32(141))},
	{V(float32(142)), V(uint64(142))},
	{V(uint64(143)), V(float64(143))},
	{V(float64(144)), V(uint64(144))},
	{V(int(145)), V(int(145))},
	{V(int(146)), V(uint(146))},
	{V(uint(147)), V(int(147))},
	{V(int(148)), V(uintptr(148))},
	{V(uintptr(149)), V(int(149))},
	{V(int(150)), V(float32(150))},
	{V(float32(151)), V(int(151))},
	{V(int(152)), V(float64(152))},
	{V(float64(153)), V(int(153))},
	{V(uint(154)), V(uint(154))},
	{V(uint(155)), V(uintptr(155))},
	{V(uintptr(156)), V(uint(156))},
	{V(uint(157)), V(float32(157))},
	{V(float32(158)), V(uint(158))},
	{V(uint(159)), V(float64(159))},
	{V(float64(160)), V(uint(160))},
	{V(uintptr(161)), V(uintptr(161))},
	{V(uintptr(162)), V(float32(162))},
	{V(float32(163)), V(uintptr(163))},
	{V(uintptr(164)), V(float64(164))},
	{V(float64(165)), V(uintptr(165))},
	{V(float32(166)), V(float32(166))},
	{V(float32(167)), V(float64(167))},
	{V(float64(168)), V(float32(168))},
	{V(float64(169)), V(float64(169))},

	// truncation
	{V(float64(1.5)), V(int(1))},

	// complex
	{V(complex64(1i)), V(complex64(1i))},
	{V(complex64(2i)), V(complex128(2i))},
	{V(complex128(3i)), V(complex64(3i))},
	{V(complex128(4i)), V(complex128(4i))},

	// string
	{V(string("hello")), V(string("hello"))},
	{V(string("bytes1")), V([]byte("bytes1"))},
	{V([]byte("bytes2")), V(string("bytes2"))},
	{V([]byte("bytes3")), V([]byte("bytes3"))},
	{V(string("runes♝")), V([]rune("runes♝"))},
	{V([]rune("runes♕")), V(string("runes♕"))},
	{V([]rune("runes🙈🙉🙊")), V([]rune("runes🙈🙉🙊"))},
	{V(int('a')), V(string("a"))},
	{V(int8('a')), V(string("a"))},
	{V(int16('a')), V(string("a"))},
	{V(int32('a')), V(string("a"))},
	{V(int64('a')), V(string("a"))},
	{V(uint('a')), V(string("a"))},
	{V(uint8('a')), V(string("a"))},
	{V(uint16('a')), V(string("a"))},
	{V(uint32('a')), V(string("a"))},
	{V(uint64('a')), V(string("a"))},
	{V(uintptr('a')), V(string("a"))},
	{V(int(-1)), V(string("\uFFFD"))},
	{V(int8(-2)), V(string("\uFFFD"))},
	{V(int16(-3)), V(string("\uFFFD"))},
	{V(int32(-4)), V(string("\uFFFD"))},
	{V(int64(-5)), V(string("\uFFFD"))},
	{V(uint(0x110001)), V(string("\uFFFD"))},
	{V(uint32(0x110002)), V(string("\uFFFD"))},
	{V(uint64(0x110003)), V(string("\uFFFD"))},
	{V(uintptr(0x110004)), V(string("\uFFFD"))},

	// named string
	{V(MyString("hello")), V(string("hello"))},
	{V(string("hello")), V(MyString("hello"))},
	{V(string("hello")), V(string("hello"))},
	{V(MyString("hello")), V(MyString("hello"))},
	{V(MyString("bytes1")), V([]byte("bytes1"))},
	{V([]byte("bytes2")), V(MyString("bytes2"))},
	{V([]byte("bytes3")), V([]byte("bytes3"))},
	{V(MyString("runes♝")), V([]rune("runes♝"))},
	{V([]rune("runes♕")), V(MyString("runes♕"))},
	{V([]rune("runes🙈🙉🙊")), V([]rune("runes🙈🙉🙊"))},
	{V([]rune("runes🙈🙉🙊")), V(MyRunes("runes🙈🙉🙊"))},
	{V(MyRunes("runes🙈🙉🙊")), V([]rune("runes🙈🙉🙊"))},
	{V(int('a')), V(MyString("a"))},
	{V(int8('a')), V(MyString("a"))},
	{V(int16('a')), V(MyString("a"))},
	{V(int32('a')), V(MyString("a"))},
	{V(int64('a')), V(MyString("a"))},
	{V(uint('a')), V(MyString("a"))},
	{V(uint8('a')), V(MyString("a"))},
	{V(uint16('a')), V(MyString("a"))},
	{V(uint32('a')), V(MyString("a"))},
	{V(uint64('a')), V(MyString("a"))},
	{V(uintptr('a')), V(MyString("a"))},
	{V(int(-1)), V(MyString("\uFFFD"))},
	{V(int8(-2)), V(MyString("\uFFFD"))},
	{V(int16(-3)), V(MyString("\uFFFD"))},
	{V(int32(-4)), V(MyString("\uFFFD"))},
	{V(int64(-5)), V(MyString("\uFFFD"))},
	{V(uint(0x110001)), V(MyString("\uFFFD"))},
	{V(uint32(0x110002)), V(MyString("\uFFFD"))},
	{V(uint64(0x110003)), V(MyString("\uFFFD"))},
	{V(uintptr(0x110004)), V(MyString("\uFFFD"))},

	// named []byte
	{V(string("bytes1")), V(MyBytes("bytes1"))},
	{V(MyBytes("bytes2")), V(string("bytes2"))},
	{V(MyBytes("bytes3")), V(MyBytes("bytes3"))},
	{V(MyString("bytes1")), V(MyBytes("bytes1"))},
	{V(MyBytes("bytes2")), V(MyString("bytes2"))},

	// named []rune
	{V(string("runes♝")), V(MyRunes("runes♝"))},
	{V(MyRunes("runes♕")), V(string("runes♕"))},
	{V(MyRunes("runes🙈🙉🙊")), V(MyRunes("runes🙈🙉🙊"))},
	{V(MyString("runes♝")), V(MyRunes("runes♝"))},
	{V(MyRunes("runes♕")), V(MyString("runes♕"))},

	// named types and equal underlying types
	{V(new(int)), V(new(integer))},
	{V(new(integer)), V(new(int))},
	{V(Empty{}), V(struct{}{})},
	{V(new(Empty)), V(new(struct{}))},
	{V(struct{}{}), V(Empty{})},
	{V(new(struct{})), V(new(Empty))},
	{V(Empty{}), V(Empty{})},
	{V(MyBytes{}), V([]byte{})},
	{V([]byte{}), V(MyBytes{})},
	{V((func())(nil)), V(MyFunc(nil))},
	{V((MyFunc)(nil)), V((func())(nil))},

	// can convert *byte and *MyByte
	{V((*byte)(nil)), V((*MyByte)(nil))},
	{V((*MyByte)(nil)), V((*byte)(nil))},

	// cannot convert mismatched array sizes
	{V([2]byte{}), V([2]byte{})},
	{V([3]byte{}), V([3]byte{})},

	// cannot convert other instances
	{V((**byte)(nil)), V((**byte)(nil))},
	{V((**MyByte)(nil)), V((**MyByte)(nil))},
	{V((chan byte)(nil)), V((chan byte)(nil))},
	{V((chan MyByte)(nil)), V((chan MyByte)(nil))},
	{V(([]byte)(nil)), V(([]byte)(nil))},
	{V(([]MyByte)(nil)), V(([]MyByte)(nil))},
	{V((map[int]byte)(nil)), V((map[int]byte)(nil))},
	{V((map[int]MyByte)(nil)), V((map[int]MyByte)(nil))},
	{V((map[byte]int)(nil)), V((map[byte]int)(nil))},
	{V((map[MyByte]int)(nil)), V((map[MyByte]int)(nil))},
	{V([2]byte{}), V([2]byte{})},
	{V([2]MyByte{}), V([2]MyByte{})},

	// other
	{V((***int)(nil)), V((***int)(nil))},
	{V((***byte)(nil)), V((***byte)(nil))},
	{V((***int32)(nil)), V((***int32)(nil))},
	{V((***int64)(nil)), V((***int64)(nil))},
	{V((chan int)(nil)), V((<-chan int)(nil))},
	{V((chan int)(nil)), V((chan<- int)(nil))},
	{V((chan string)(nil)), V((<-chan string)(nil))},
	{V((chan string)(nil)), V((chan<- string)(nil))},
	{V((chan byte)(nil)), V((chan byte)(nil))},
	{V((chan MyByte)(nil)), V((chan MyByte)(nil))},
	{V((map[int]bool)(nil)), V((map[int]bool)(nil))},
	{V((map[int]byte)(nil)), V((map[int]byte)(nil))},
	{V((map[uint]bool)(nil)), V((map[uint]bool)(nil))},
	{V([]uint(nil)), V([]uint(nil))},
	{V([]int(nil)), V([]int(nil))},
	{V(new(interface{})), V(new(interface{}))},
	{V(new(io.Reader)), V(new(io.Reader))},
	{V(new(io.Writer)), V(new(io.Writer))},

	// interfaces
	{V(int(1)), EmptyInterfaceV(int(1))},
	{V(string("hello")), EmptyInterfaceV(string("hello"))},
	{V(new(bytes.Buffer)), ReaderV(new(bytes.Buffer))},
	{ReadWriterV(new(bytes.Buffer)), ReaderV(new(bytes.Buffer))},
	{V(new(bytes.Buffer)), ReadWriterV(new(bytes.Buffer))},
}

func TestConvert(t *testing.T) {
	canConvert := map[[2]Type]bool{}
	all := map[Type]bool{}

	for _, tt := range convertTests {
		t1 := tt.in.Type()
		if !t1.ConvertibleTo(t1) {
			t.Errorf("(%s).ConvertibleTo(%s) = false, want true", t1, t1)
			continue
		}

		t2 := tt.out.Type()
		if !t1.ConvertibleTo(t2) {
			t.Errorf("(%s).ConvertibleTo(%s) = false, want true", t1, t2)
			continue
		}

		all[t1] = true
		all[t2] = true
		canConvert[[2]Type{t1, t2}] = true

		// vout1 represents the in value converted to the in type.
		v1 := tt.in
		vout1 := v1.Convert(t1)
		out1 := vout1.Interface()
		if vout1.Type() != tt.in.Type() || !DeepEqual(out1, tt.in.Interface()) {
			t.Errorf("ValueOf(%T(%[1]v)).Convert(%s) = %T(%[3]v), want %T(%[4]v)", tt.in.Interface(), t1, out1, tt.in.Interface())
		}

		// vout2 represents the in value converted to the out type.
		vout2 := v1.Convert(t2)
		out2 := vout2.Interface()
		if vout2.Type() != tt.out.Type() || !DeepEqual(out2, tt.out.Interface()) {
			t.Errorf("ValueOf(%T(%[1]v)).Convert(%s) = %T(%[3]v), want %T(%[4]v)", tt.in.Interface(), t2, out2, tt.out.Interface())
		}

		// vout3 represents a new value of the out type, set to vout2.  This makes
		// sure the converted value vout2 is really usable as a regular value.
		vout3 := New(t2).Elem()
		vout3.Set(vout2)
		out3 := vout3.Interface()
		if vout3.Type() != tt.out.Type() || !DeepEqual(out3, tt.out.Interface()) {
			t.Errorf("Set(ValueOf(%T(%[1]v)).Convert(%s)) = %T(%[3]v), want %T(%[4]v)", tt.in.Interface(), t2, out3, tt.out.Interface())
		}

		if IsRO(v1) {
			t.Errorf("table entry %v is RO, should not be", v1)
		}
		if IsRO(vout1) {
			t.Errorf("self-conversion output %v is RO, should not be", vout1)
		}
		if IsRO(vout2) {
			t.Errorf("conversion output %v is RO, should not be", vout2)
		}
		if IsRO(vout3) {
			t.Errorf("set(conversion output) %v is RO, should not be", vout3)
		}
		if !IsRO(MakeRO(v1).Convert(t1)) {
			t.Errorf("RO self-conversion output %v is not RO, should be", v1)
		}
		if !IsRO(MakeRO(v1).Convert(t2)) {
			t.Errorf("RO conversion output %v is not RO, should be", v1)
		}
	}

	// Assume that of all the types we saw during the tests,
	// if there wasn't an explicit entry for a conversion between
	// a pair of types, then it's not to be allowed. This checks for
	// things like 'int64' converting to '*int'.
	for t1 := range all {
		for t2 := range all {
			expectOK := t1 == t2 || canConvert[[2]Type{t1, t2}] || t2.Kind() == Interface && t2.NumMethod() == 0
			if ok := t1.ConvertibleTo(t2); ok != expectOK {
				t.Errorf("(%s).ConvertibleTo(%s) = %v, want %v", t1, t2, ok, expectOK)
			}
		}
	}
}

type ComparableStruct struct {
	X int
}

type NonComparableStruct struct {
	X int
	Y map[string]int
}

var comparableTests = []struct {
	typ Type
	ok  bool
}{
	{TypeOf(1), true},
	{TypeOf("hello"), true},
	{TypeOf(new(byte)), true},
	{TypeOf((func())(nil)), false},
	{TypeOf([]byte{}), false},
	{TypeOf(map[string]int{}), false},
	{TypeOf(make(chan int)), true},
	{TypeOf(1.5), true},
	{TypeOf(false), true},
	{TypeOf(1i), true},
	{TypeOf(ComparableStruct{}), true},
	{TypeOf(NonComparableStruct{}), false},
	{TypeOf([10]map[string]int{}), false},
	{TypeOf([10]string{}), true},
	{TypeOf(new(interface{})).Elem(), true},
}

func TestComparable(t *testing.T) {
	for _, tt := range comparableTests {
		if ok := tt.typ.Comparable(); ok != tt.ok {
			t.Errorf("TypeOf(%v).Comparable() = %v, want %v", tt.typ, ok, tt.ok)
		}
	}
}

func TestOverflow(t *testing.T) {
	if ovf := V(float64(0)).OverflowFloat(1e300); ovf {
		t.Errorf("%v wrongly overflows float64", 1e300)
	}

	maxFloat32 := float64((1<<24 - 1) << (127 - 23))
	if ovf := V(float32(0)).OverflowFloat(maxFloat32); ovf {
		t.Errorf("%v wrongly overflows float32", maxFloat32)
	}
	ovfFloat32 := float64((1<<24-1)<<(127-23) + 1<<(127-52))
	if ovf := V(float32(0)).OverflowFloat(ovfFloat32); !ovf {
		t.Errorf("%v should overflow float32", ovfFloat32)
	}
	if ovf := V(float32(0)).OverflowFloat(-ovfFloat32); !ovf {
		t.Errorf("%v should overflow float32", -ovfFloat32)
	}

	maxInt32 := int64(0x7fffffff)
	if ovf := V(int32(0)).OverflowInt(maxInt32); ovf {
		t.Errorf("%v wrongly overflows int32", maxInt32)
	}
	if ovf := V(int32(0)).OverflowInt(-1 << 31); ovf {
		t.Errorf("%v wrongly overflows int32", -int64(1)<<31)
	}
	ovfInt32 := int64(1 << 31)
	if ovf := V(int32(0)).OverflowInt(ovfInt32); !ovf {
		t.Errorf("%v should overflow int32", ovfInt32)
	}

	maxUint32 := uint64(0xffffffff)
	if ovf := V(uint32(0)).OverflowUint(maxUint32); ovf {
		t.Errorf("%v wrongly overflows uint32", maxUint32)
	}
	ovfUint32 := uint64(1 << 32)
	if ovf := V(uint32(0)).OverflowUint(ovfUint32); !ovf {
		t.Errorf("%v should overflow uint32", ovfUint32)
	}
}

func checkSameType(t *testing.T, x, y interface{}) {
	if TypeOf(x) != TypeOf(y) {
		t.Errorf("did not find preexisting type for %s (vs %s)", TypeOf(x), TypeOf(y))
	}
}

func TestArrayOf(t *testing.T) {
	// TODO(rsc): Finish ArrayOf and enable-test.
	t.Skip("ArrayOf is not finished (and not exported)")

	// check construction and use of type not in binary
	type T int
	at := ArrayOf(10, TypeOf(T(1)))
	v := New(at).Elem()
	for i := 0; i < v.Len(); i++ {
		v.Index(i).Set(ValueOf(T(i)))
	}
	s := fmt.Sprint(v.Interface())
	want := "[0 1 2 3 4 5 6 7 8 9]"
	if s != want {
		t.Errorf("constructed array = %s, want %s", s, want)
	}

	// check that type already in binary is found
	checkSameType(t, Zero(ArrayOf(5, TypeOf(T(1)))).Interface(), [5]T{})
}

func TestSliceOf(t *testing.T) {
	// check construction and use of type not in binary
	type T int
	st := SliceOf(TypeOf(T(1)))
	v := MakeSlice(st, 10, 10)
	runtime.GC()
	for i := 0; i < v.Len(); i++ {
		v.Index(i).Set(ValueOf(T(i)))
		runtime.GC()
	}
	s := fmt.Sprint(v.Interface())
	want := "[0 1 2 3 4 5 6 7 8 9]"
	if s != want {
		t.Errorf("constructed slice = %s, want %s", s, want)
	}

	// check that type already in binary is found
	type T1 int
	checkSameType(t, Zero(SliceOf(TypeOf(T1(1)))).Interface(), []T1{})
}

func TestSliceOverflow(t *testing.T) {
	// check that MakeSlice panics when size of slice overflows uint
	const S = 1e6
	s := uint(S)
	l := (1<<(unsafe.Sizeof((*byte)(nil))*8)-1)/s + 1
	if l*s >= s {
		t.Fatal("slice size does not overflow")
	}
	var x [S]byte
	st := SliceOf(TypeOf(x))
	defer func() {
		err := recover()
		if err == nil {
			t.Fatal("slice overflow does not panic")
		}
	}()
	MakeSlice(st, int(l), int(l))
}

func TestSliceOfGC(t *testing.T) {
	type T *uintptr
	tt := TypeOf(T(nil))
	st := SliceOf(tt)
	const n = 100
	var x []interface{}
	for i := 0; i < n; i++ {
		v := MakeSlice(st, n, n)
		for j := 0; j < v.Len(); j++ {
			p := new(uintptr)
			*p = uintptr(i*n + j)
			v.Index(j).Set(ValueOf(p).Convert(tt))
		}
		x = append(x, v.Interface())
	}
	runtime.GC()

	for i, xi := range x {
		v := ValueOf(xi)
		for j := 0; j < v.Len(); j++ {
			k := v.Index(j).Elem().Interface()
			if k != uintptr(i*n+j) {
				t.Errorf("lost x[%d][%d] = %d, want %d", i, j, k, i*n+j)
			}
		}
	}
}

func TestChanOf(t *testing.T) {
	// check construction and use of type not in binary
	type T string
	ct := ChanOf(BothDir, TypeOf(T("")))
	v := MakeChan(ct, 2)
	runtime.GC()
	v.Send(ValueOf(T("hello")))
	runtime.GC()
	v.Send(ValueOf(T("world")))
	runtime.GC()

	sv1, _ := v.Recv()
	sv2, _ := v.Recv()
	s1 := sv1.String()
	s2 := sv2.String()
	if s1 != "hello" || s2 != "world" {
		t.Errorf("constructed chan: have %q, %q, want %q, %q", s1, s2, "hello", "world")
	}

	// check that type already in binary is found
	type T1 int
	checkSameType(t, Zero(ChanOf(BothDir, TypeOf(T1(1)))).Interface(), (chan T1)(nil))
}

func TestChanOfGC(t *testing.T) {
	done := make(chan bool, 1)
	go func() {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			panic("deadlock in TestChanOfGC")
		}
	}()

	defer func() {
		done <- true
	}()

	type T *uintptr
	tt := TypeOf(T(nil))
	ct := ChanOf(BothDir, tt)

	// NOTE: The garbage collector handles allocated channels specially,
	// so we have to save pointers to channels in x; the pointer code will
	// use the gc info in the newly constructed chan type.
	const n = 100
	var x []interface{}
	for i := 0; i < n; i++ {
		v := MakeChan(ct, n)
		for j := 0; j < n; j++ {
			p := new(uintptr)
			*p = uintptr(i*n + j)
			v.Send(ValueOf(p).Convert(tt))
		}
		pv := New(ct)
		pv.Elem().Set(v)
		x = append(x, pv.Interface())
	}
	runtime.GC()

	for i, xi := range x {
		v := ValueOf(xi).Elem()
		for j := 0; j < n; j++ {
			pv, _ := v.Recv()
			k := pv.Elem().Interface()
			if k != uintptr(i*n+j) {
				t.Errorf("lost x[%d][%d] = %d, want %d", i, j, k, i*n+j)
			}
		}
	}
}

func TestMapOf(t *testing.T) {
	// check construction and use of type not in binary
	type K string
	type V float64

	v := MakeMap(MapOf(TypeOf(K("")), TypeOf(V(0))))
	runtime.GC()
	v.SetMapIndex(ValueOf(K("a")), ValueOf(V(1)))
	runtime.GC()

	s := fmt.Sprint(v.Interface())
	want := "map[a:1]"
	if s != want {
		t.Errorf("constructed map = %s, want %s", s, want)
	}

	// check that type already in binary is found
	checkSameType(t, Zero(MapOf(TypeOf(V(0)), TypeOf(K("")))).Interface(), map[V]K(nil))

	// check that invalid key type panics
	shouldPanic(func() { MapOf(TypeOf((func())(nil)), TypeOf(false)) })
}

func TestMapOfGCKeys(t *testing.T) {
	type T *uintptr
	tt := TypeOf(T(nil))
	mt := MapOf(tt, TypeOf(false))

	// NOTE: The garbage collector handles allocated maps specially,
	// so we have to save pointers to maps in x; the pointer code will
	// use the gc info in the newly constructed map type.
	const n = 100
	var x []interface{}
	for i := 0; i < n; i++ {
		v := MakeMap(mt)
		for j := 0; j < n; j++ {
			p := new(uintptr)
			*p = uintptr(i*n + j)
			v.SetMapIndex(ValueOf(p).Convert(tt), ValueOf(true))
		}
		pv := New(mt)
		pv.Elem().Set(v)
		x = append(x, pv.Interface())
	}
	runtime.GC()

	for i, xi := range x {
		v := ValueOf(xi).Elem()
		var out []int
		for _, kv := range v.MapKeys() {
			out = append(out, int(kv.Elem().Interface().(uintptr)))
		}
		sort.Ints(out)
		for j, k := range out {
			if k != i*n+j {
				t.Errorf("lost x[%d][%d] = %d, want %d", i, j, k, i*n+j)
			}
		}
	}
}

func TestMapOfGCValues(t *testing.T) {
	type T *uintptr
	tt := TypeOf(T(nil))
	mt := MapOf(TypeOf(1), tt)

	// NOTE: The garbage collector handles allocated maps specially,
	// so we have to save pointers to maps in x; the pointer code will
	// use the gc info in the newly constructed map type.
	const n = 100
	var x []interface{}
	for i := 0; i < n; i++ {
		v := MakeMap(mt)
		for j := 0; j < n; j++ {
			p := new(uintptr)
			*p = uintptr(i*n + j)
			v.SetMapIndex(ValueOf(j), ValueOf(p).Convert(tt))
		}
		pv := New(mt)
		pv.Elem().Set(v)
		x = append(x, pv.Interface())
	}
	runtime.GC()

	for i, xi := range x {
		v := ValueOf(xi).Elem()
		for j := 0; j < n; j++ {
			k := v.MapIndex(ValueOf(j)).Elem().Interface().(uintptr)
			if k != uintptr(i*n+j) {
				t.Errorf("lost x[%d][%d] = %d, want %d", i, j, k, i*n+j)
			}
		}
	}
}

type B1 struct {
	X int
	Y int
	Z int
}

func BenchmarkFieldByName1(b *testing.B) {
	t := TypeOf(B1{})
	for i := 0; i < b.N; i++ {
		t.FieldByName("Z")
	}
}

func BenchmarkFieldByName2(b *testing.B) {
	t := TypeOf(S3{})
	for i := 0; i < b.N; i++ {
		t.FieldByName("B")
	}
}

type R0 struct {
	*R1
	*R2
	*R3
	*R4
}

type R1 struct {
	*R5
	*R6
	*R7
	*R8
}

type R2 R1
type R3 R1
type R4 R1

type R5 struct {
	*R9
	*R10
	*R11
	*R12
}

type R6 R5
type R7 R5
type R8 R5

type R9 struct {
	*R13
	*R14
	*R15
	*R16
}

type R10 R9
type R11 R9
type R12 R9

type R13 struct {
	*R17
	*R18
	*R19
	*R20
}

type R14 R13
type R15 R13
type R16 R13

type R17 struct {
	*R21
	*R22
	*R23
	*R24
}

type R18 R17
type R19 R17
type R20 R17

type R21 struct {
	X int
}

type R22 R21
type R23 R21
type R24 R21

func TestEmbed(t *testing.T) {
	typ := TypeOf(R0{})
	f, ok := typ.FieldByName("X")
	if ok {
		t.Fatalf(`FieldByName("X") should fail, returned %v`, f.Index)
	}
}

func BenchmarkFieldByName3(b *testing.B) {
	t := TypeOf(R0{})
	for i := 0; i < b.N; i++ {
		t.FieldByName("X")
	}
}

type S struct {
	i1 int64
	i2 int64
}

func BenchmarkInterfaceBig(b *testing.B) {
	v := ValueOf(S{})
	for i := 0; i < b.N; i++ {
		v.Interface()
	}
	b.StopTimer()
}

func TestAllocsInterfaceBig(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	v := ValueOf(S{})
	if allocs := testing.AllocsPerRun(100, func() { v.Interface() }); allocs > 0 {
		t.Error("allocs:", allocs)
	}
}

func BenchmarkInterfaceSmall(b *testing.B) {
	v := ValueOf(int64(0))
	for i := 0; i < b.N; i++ {
		v.Interface()
	}
}

func TestAllocsInterfaceSmall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	v := ValueOf(int64(0))
	if allocs := testing.AllocsPerRun(100, func() { v.Interface() }); allocs > 0 {
		t.Error("allocs:", allocs)
	}
}

// An exhaustive is a mechanism for writing exhaustive or stochastic tests.
// The basic usage is:
//
//	for x.Next() {
//		... code using x.Maybe() or x.Choice(n) to create test cases ...
//	}
//
// Each iteration of the loop returns a different set of results, until all
// possible result sets have been explored. It is okay for different code paths
// to make different method call sequences on x, but there must be no
// other source of non-determinism in the call sequences.
//
// When faced with a new decision, x chooses randomly. Future explorations
// of that path will choose successive values for the result. Thus, stopping
// the loop after a fixed number of iterations gives somewhat stochastic
// testing.
//
// Example:
//
//	for x.Next() {
//		v := make([]bool, x.Choose(4))
//		for i := range v {
//			v[i] = x.Maybe()
//		}
//		fmt.Println(v)
//	}
//
// prints (in some order):
//
//	[]
//	[false]
//	[true]
//	[false false]
//	[false true]
//	...
//	[true true]
//	[false false false]
//	...
//	[true true true]
//	[false false false false]
//	...
//	[true true true true]
//
type exhaustive struct {
	r    *rand.Rand
	pos  int
	last []choice
}

type choice struct {
	off int
	n   int
	max int
}

func (x *exhaustive) Next() bool {
	if x.r == nil {
		x.r = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	x.pos = 0
	if x.last == nil {
		x.last = []choice{}
		return true
	}
	for i := len(x.last) - 1; i >= 0; i-- {
		c := &x.last[i]
		if c.n+1 < c.max {
			c.n++
			x.last = x.last[:i+1]
			return true
		}
	}
	return false
}

func (x *exhaustive) Choose(max int) int {
	if x.pos >= len(x.last) {
		x.last = append(x.last, choice{x.r.Intn(max), 0, max})
	}
	c := &x.last[x.pos]
	x.pos++
	if c.max != max {
		panic("inconsistent use of exhaustive tester")
	}
	return (c.n + c.off) % max
}

func (x *exhaustive) Maybe() bool {
	return x.Choose(2) == 1
}

func GCFunc(args []Value) []Value {
	runtime.GC()
	return []Value{}
}

func TestReflectFuncTraceback(t *testing.T) {
	f := MakeFunc(TypeOf(func() {}), GCFunc)
	f.Call([]Value{})
}

func TestReflectMethodTraceback(t *testing.T) {
	p := Point{3, 4}
	m := ValueOf(p).MethodByName("GCMethod")
	i := ValueOf(m.Interface()).Call([]Value{ValueOf(5)})[0].Int()
	if i != 8 {
		t.Errorf("Call returned %d; want 8", i)
	}
}

func TestBigZero(t *testing.T) {
	const size = 1 << 10
	var v [size]byte
	z := Zero(ValueOf(v).Type()).Interface().([size]byte)
	for i := 0; i < size; i++ {
		if z[i] != 0 {
			t.Fatalf("Zero object not all zero, index %d", i)
		}
	}
}

func TestFieldByIndexNil(t *testing.T) {
	type P struct {
		F int
	}
	type T struct {
		*P
	}
	v := ValueOf(T{})

	v.FieldByName("P") // should be fine

	defer func() {
		if err := recover(); err == nil {
			t.Fatalf("no error")
		} else if !strings.Contains(fmt.Sprint(err), "nil pointer to embedded struct") {
			t.Fatalf(`err=%q, wanted error containing "nil pointer to embedded struct"`, err)
		}
	}()
	v.FieldByName("F") // should panic

	t.Fatalf("did not panic")
}

// Given
//	type Outer struct {
//		*Inner
//		...
//	}
// the compiler generates the implementation of (*Outer).M dispatching to the embedded Inner.
// The implementation is logically:
//	func (p *Outer) M() {
//		(p.Inner).M()
//	}
// but since the only change here is the replacement of one pointer receiver with another,
// the actual generated code overwrites the original receiver with the p.Inner pointer and
// then jumps to the M method expecting the *Inner receiver.
//
// During reflect.Value.Call, we create an argument frame and the associated data structures
// to describe it to the garbage collector, populate the frame, call reflect.call to
// run a function call using that frame, and then copy the results back out of the frame.
// The reflect.call function does a memmove of the frame structure onto the
// stack (to set up the inputs), runs the call, and the memmoves the stack back to
// the frame structure (to preserve the outputs).
//
// Originally reflect.call did not distinguish inputs from outputs: both memmoves
// were for the full stack frame. However, in the case where the called function was
// one of these wrappers, the rewritten receiver is almost certainly a different type
// than the original receiver. This is not a problem on the stack, where we use the
// program counter to determine the type information and understand that
// during (*Outer).M the receiver is an *Outer while during (*Inner).M the receiver in the same
// memory word is now an *Inner. But in the statically typed argument frame created
// by reflect, the receiver is always an *Outer. Copying the modified receiver pointer
// off the stack into the frame will store an *Inner there, and then if a garbage collection
// happens to scan that argument frame before it is discarded, it will scan the *Inner
// memory as if it were an *Outer. If the two have different memory layouts, the
// collection will intepret the memory incorrectly.
//
// One such possible incorrect interpretation is to treat two arbitrary memory words
// (Inner.P1 and Inner.P2 below) as an interface (Outer.R below). Because interpreting
// an interface requires dereferencing the itab word, the misinterpretation will try to
// deference Inner.P1, causing a crash during garbage collection.
//
// This came up in a real program in issue 7725.

type Outer struct {
	*Inner
	R io.Reader
}

type Inner struct {
	X  *Outer
	P1 uintptr
	P2 uintptr
}

func (pi *Inner) M() {
	// Clear references to pi so that the only way the
	// garbage collection will find the pointer is in the
	// argument frame, typed as a *Outer.
	pi.X.Inner = nil

	// Set up an interface value that will cause a crash.
	// P1 = 1 is a non-zero, so the interface looks non-nil.
	// P2 = pi ensures that the data word points into the
	// allocated heap; if not the collection skips the interface
	// value as irrelevant, without dereferencing P1.
	pi.P1 = 1
	pi.P2 = uintptr(unsafe.Pointer(pi))
}

func TestCallMethodJump(t *testing.T) {
	// In reflect.Value.Call, trigger a garbage collection after reflect.call
	// returns but before the args frame has been discarded.
	// This is a little clumsy but makes the failure repeatable.
	*CallGC = true

	p := &Outer{Inner: new(Inner)}
	p.Inner.X = p
	ValueOf(p).Method(0).Call(nil)

	// Stop garbage collecting during reflect.call.
	*CallGC = false
}

func TestMakeFuncStackCopy(t *testing.T) {
	target := func(in []Value) []Value {
		runtime.GC()
		useStack(16)
		return []Value{ValueOf(9)}
	}

	var concrete func(*int, int) int
	fn := MakeFunc(ValueOf(concrete).Type(), target)
	ValueOf(&concrete).Elem().Set(fn)
	x := concrete(nil, 7)
	if x != 9 {
		t.Errorf("have %#q want 9", x)
	}
}

// use about n KB of stack
func useStack(n int) {
	if n == 0 {
		return
	}
	var b [1024]byte // makes frame about 1KB
	useStack(n - 1 + int(b[99]))
}

type Impl struct{}

func (Impl) f() {}

func TestValueString(t *testing.T) {
	rv := ValueOf(Impl{})
	if rv.String() != "<reflect_test.Impl Value>" {
		t.Errorf("ValueOf(Impl{}).String() = %q, want %q", rv.String(), "<reflect_test.Impl Value>")
	}

	method := rv.Method(0)
	if method.String() != "<func() Value>" {
		t.Errorf("ValueOf(Impl{}).Method(0).String() = %q, want %q", method.String(), "<func() Value>")
	}
}

func TestInvalid(t *testing.T) {
	// Used to have inconsistency between IsValid() and Kind() != Invalid.
	type T struct{ v interface{} }

	v := ValueOf(T{}).Field(0)
	if v.IsValid() != true || v.Kind() != Interface {
		t.Errorf("field: IsValid=%v, Kind=%v, want true, Interface", v.IsValid(), v.Kind())
	}
	v = v.Elem()
	if v.IsValid() != false || v.Kind() != Invalid {
		t.Errorf("field elem: IsValid=%v, Kind=%v, want false, Invalid", v.IsValid(), v.Kind())
	}
}

// Issue 8917.
func TestLargeGCProg(t *testing.T) {
	fv := ValueOf(func([256]*byte) {})
	fv.Call([]Value{ValueOf([256]*byte{})})
}

// Issue 9179.
func TestCallGC(t *testing.T) {
	f := func(a, b, c, d, e string) {
	}
	g := func(in []Value) []Value {
		runtime.GC()
		return nil
	}
	typ := ValueOf(f).Type()
	f2 := MakeFunc(typ, g).Interface().(func(string, string, string, string, string))
	f2("four", "five5", "six666", "seven77", "eight888")
}

type funcLayoutTest struct {
	rcvr, t            Type
	argsize, retOffset uintptr
	stack              []byte
}

var funcLayoutTests []funcLayoutTest

func init() {
	var argAlign = PtrSize
	if runtime.GOARCH == "amd64p32" {
		argAlign = 2 * PtrSize
	}
	roundup := func(x uintptr, a uintptr) uintptr {
		return (x + a - 1) / a * a
	}

	funcLayoutTests = append(funcLayoutTests,
		funcLayoutTest{
			nil,
			ValueOf(func(a, b string) string { return "" }).Type(),
			4 * PtrSize,
			4 * PtrSize,
			[]byte{BitsPointer, BitsScalar, BitsPointer},
		})

	var r []byte
	if PtrSize == 4 {
		r = []byte{BitsScalar, BitsScalar, BitsScalar, BitsPointer}
	} else {
		r = []byte{BitsScalar, BitsScalar, BitsPointer}
	}
	funcLayoutTests = append(funcLayoutTests,
		funcLayoutTest{
			nil,
			ValueOf(func(a, b, c uint32, p *byte, d uint16) {}).Type(),
			roundup(3*4, PtrSize) + PtrSize + 2,
			roundup(roundup(3*4, PtrSize)+PtrSize+2, argAlign),
			r,
		})

	funcLayoutTests = append(funcLayoutTests,
		funcLayoutTest{
			nil,
			ValueOf(func(a map[int]int, b uintptr, c interface{}) {}).Type(),
			4 * PtrSize,
			4 * PtrSize,
			[]byte{BitsPointer, BitsScalar, BitsPointer, BitsPointer},
		})

	type S struct {
		a, b uintptr
		c, d *byte
	}
	funcLayoutTests = append(funcLayoutTests,
		funcLayoutTest{
			nil,
			ValueOf(func(a S) {}).Type(),
			4 * PtrSize,
			4 * PtrSize,
			[]byte{BitsScalar, BitsScalar, BitsPointer, BitsPointer},
		})

	funcLayoutTests = append(funcLayoutTests,
		funcLayoutTest{
			ValueOf((*byte)(nil)).Type(),
			ValueOf(func(a uintptr, b *int) {}).Type(),
			3 * PtrSize,
			roundup(3*PtrSize, argAlign),
			[]byte{BitsPointer, BitsScalar, BitsPointer},
		})
}

func TestFuncLayout(t *testing.T) {
	for _, lt := range funcLayoutTests {
		_, argsize, retOffset, stack := FuncLayout(lt.t, lt.rcvr)
		if argsize != lt.argsize {
			t.Errorf("funcLayout(%v, %v).argsize=%d, want %d", lt.t, lt.rcvr, argsize, lt.argsize)
		}
		if retOffset != lt.retOffset {
			t.Errorf("funcLayout(%v, %v).retOffset=%d, want %d", lt.t, lt.rcvr, retOffset, lt.retOffset)
		}
		if !bytes.Equal(stack, lt.stack) {
			t.Errorf("funcLayout(%v, %v).stack=%v, want %v", lt.t, lt.rcvr, stack, lt.stack)
		}
	}
}
