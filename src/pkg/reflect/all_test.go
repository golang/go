// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect_test

import (
	"container/vector"
	"io"
	"os"
	. "reflect"
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
	pair{struct{ x int }{}, "int"},
	pair{struct{ x int8 }{}, "int8"},
	pair{struct{ x int16 }{}, "int16"},
	pair{struct{ x int32 }{}, "int32"},
	pair{struct{ x int64 }{}, "int64"},
	pair{struct{ x uint }{}, "uint"},
	pair{struct{ x uint8 }{}, "uint8"},
	pair{struct{ x uint16 }{}, "uint16"},
	pair{struct{ x uint32 }{}, "uint32"},
	pair{struct{ x uint64 }{}, "uint64"},
	pair{struct{ x float }{}, "float"},
	pair{struct{ x float32 }{}, "float32"},
	pair{struct{ x float64 }{}, "float64"},
	pair{struct{ x int8 }{}, "int8"},
	pair{struct{ x (**int8) }{}, "**int8"},
	pair{struct{ x (**integer) }{}, "**reflect_test.integer"},
	pair{struct{ x ([32]int32) }{}, "[32]int32"},
	pair{struct{ x ([]int8) }{}, "[]int8"},
	pair{struct{ x (map[string]int32) }{}, "map[string] int32"},
	pair{struct{ x (chan<- string) }{}, "chan<- string"},
	pair{struct {
		x struct {
			c chan *int32
			d float32
		}
	}{},
		"struct { c chan *int32; d float32 }",
	},
	pair{struct{ x (func(a int8, b int32)) }{}, "func(int8, int32)"},
	pair{struct {
		x struct {
			c func(chan *integer, *int8)
		}
	}{},
		"struct { c func(chan *reflect_test.integer, *int8) }",
	},
	pair{struct {
		x struct {
			a int8
			b int32
		}
	}{},
		"struct { a int8; b int32 }",
	},
	pair{struct {
		x struct {
			a int8
			b int8
			c int32
		}
	}{},
		"struct { a int8; b int8; c int32 }",
	},
	pair{struct {
		x struct {
			a int8
			b int8
			c int8
			d int32
		}
	}{},
		"struct { a int8; b int8; c int8; d int32 }",
	},
	pair{struct {
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
	pair{struct {
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
	pair{struct {
		x struct {
			a int8 "hi there"
		}
	}{},
		`struct { a int8 "hi there" }`,
	},
	pair{struct {
		x struct {
			a int8 "hi \x00there\t\n\"\\"
		}
	}{},
		`struct { a int8 "hi \x00there\t\n\"\\" }`,
	},
	pair{struct {
		x struct {
			f func(args ...)
		}
	}{},
		"struct { f func(...) }",
	},
	pair{struct {
		x (interface {
			a(func(func(int) int) func(func(int)) int)
			b()
		})
	}{},
		"interface { a(func(func(int) int) func(func(int)) int); b() }",
	},
}

var valueTests = []pair{
	pair{(int8)(0), "8"},
	pair{(int16)(0), "16"},
	pair{(int32)(0), "32"},
	pair{(int64)(0), "64"},
	pair{(uint8)(0), "8"},
	pair{(uint16)(0), "16"},
	pair{(uint32)(0), "32"},
	pair{(uint64)(0), "64"},
	pair{(float32)(0), "32.1"},
	pair{(float64)(0), "64.2"},
	pair{(string)(""), "stringy cheese"},
	pair{(bool)(false), "true"},
	pair{(*int8)(nil), "*int8(0)"},
	pair{(**int8)(nil), "**int8(0)"},
	pair{([5]int32){}, "[5]int32{0, 0, 0, 0, 0}"},
	pair{(**integer)(nil), "**reflect_test.integer(0)"},
	pair{(map[string]int32)(nil), "map[string] int32{<can't iterate on maps>}"},
	pair{(chan<- string)(nil), "chan<- string"},
	pair{(struct {
		c chan *int32
		d float32
	}){},
		"struct { c chan *int32; d float32 }{chan *int32, 0}",
	},
	pair{(func(a int8, b int32))(nil), "func(int8, int32)(0)"},
	pair{(struct {
		c func(chan *integer, *int8)
	}){},
		"struct { c func(chan *reflect_test.integer, *int8) }{func(chan *reflect_test.integer, *int8)(0)}",
	},
	pair{(struct {
		a int8
		b int32
	}){},
		"struct { a int8; b int32 }{0, 0}",
	},
	pair{(struct {
		a int8
		b int8
		c int32
	}){},
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
		testType(t, i, NewValue(tt.i).(*StructValue).Field(0).Type(), tt.s)
	}
}

func TestSet(t *testing.T) {
	for i, tt := range valueTests {
		v := NewValue(tt.i)
		switch v := v.(type) {
		case *IntValue:
			v.Set(132)
		case *Int8Value:
			v.Set(8)
		case *Int16Value:
			v.Set(16)
		case *Int32Value:
			v.Set(32)
		case *Int64Value:
			v.Set(64)
		case *UintValue:
			v.Set(132)
		case *Uint8Value:
			v.Set(8)
		case *Uint16Value:
			v.Set(16)
		case *Uint32Value:
			v.Set(32)
		case *Uint64Value:
			v.Set(64)
		case *FloatValue:
			v.Set(3200.0)
		case *Float32Value:
			v.Set(32.1)
		case *Float64Value:
			v.Set(64.2)
		case *StringValue:
			v.Set("stringy cheese")
		case *BoolValue:
			v.Set(true)
		}
		s := valueToString(v)
		if s != tt.s {
			t.Errorf("#%d: have %#q, want %#q", i, s, tt.s)
		}
	}
}

func TestSetValue(t *testing.T) {
	for i, tt := range valueTests {
		v := NewValue(tt.i)
		switch v := v.(type) {
		case *IntValue:
			v.SetValue(NewValue(int(132)))
		case *Int8Value:
			v.SetValue(NewValue(int8(8)))
		case *Int16Value:
			v.SetValue(NewValue(int16(16)))
		case *Int32Value:
			v.SetValue(NewValue(int32(32)))
		case *Int64Value:
			v.SetValue(NewValue(int64(64)))
		case *UintValue:
			v.SetValue(NewValue(uint(132)))
		case *Uint8Value:
			v.SetValue(NewValue(uint8(8)))
		case *Uint16Value:
			v.SetValue(NewValue(uint16(16)))
		case *Uint32Value:
			v.SetValue(NewValue(uint32(32)))
		case *Uint64Value:
			v.SetValue(NewValue(uint64(64)))
		case *FloatValue:
			v.SetValue(NewValue(float(3200.0)))
		case *Float32Value:
			v.SetValue(NewValue(float32(32.1)))
		case *Float64Value:
			v.SetValue(NewValue(float64(64.2)))
		case *StringValue:
			v.SetValue(NewValue("stringy cheese"))
		case *BoolValue:
			v.SetValue(NewValue(true))
		}
		s := valueToString(v)
		if s != tt.s {
			t.Errorf("#%d: have %#q, want %#q", i, s, tt.s)
		}
	}
}

var _i = 7

var valueToStringTests = []pair{
	pair{123, "123"},
	pair{123.4, "123.4"},
	pair{byte(123), "123"},
	pair{"abc", "abc"},
	pair{T{123, 456.75, "hello", &_i}, "reflect_test.T{123, 456.75, hello, *int(&7)}"},
	pair{new(chan *T), "*chan *reflect_test.T(&chan *reflect_test.T)"},
	pair{[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}"},
	pair{&[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "*[10]int(&[10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})"},
	pair{[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}"},
	pair{&[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, "*[]int(&[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})"},
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
	v := NewValue([10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	v.(*ArrayValue).Elem(4).(*IntValue).Set(123)
	s := valueToString(v)
	const want = "[10]int{1, 2, 3, 4, 123, 6, 7, 8, 9, 10}"
	if s != want {
		t.Errorf("[10]int: have %#q want %#q", s, want)
	}

	v = NewValue([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	v.(*SliceValue).Elem(4).(*IntValue).Set(123)
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
	vi := NewValue(i)
	vip.(*PtrValue).Elem().(*PtrValue).PointTo(vi)
	if *ip != 1234 {
		t.Errorf("got %d, want 1234", *ip)
	}
}

func TestAll(t *testing.T) {
	testType(t, 1, Typeof((int8)(0)), "int8")
	testType(t, 2, Typeof((*int8)(nil)).(*PtrType).Elem(), "int8")

	typ := Typeof((*struct {
		c chan *int32
		d float32
	})(nil))
	testType(t, 3, typ, "*struct { c chan *int32; d float32 }")
	etyp := typ.(*PtrType).Elem()
	testType(t, 4, etyp, "struct { c chan *int32; d float32 }")
	styp := etyp.(*StructType)
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
	testType(t, 8, typ.(*ArrayType).Elem(), "int32")

	typ = Typeof((map[string]*int32)(nil))
	testType(t, 9, typ, "map[string] *int32")
	mtyp := typ.(*MapType)
	testType(t, 10, mtyp.Key(), "string")
	testType(t, 11, mtyp.Elem(), "*int32")

	typ = Typeof((chan<- string)(nil))
	testType(t, 12, typ, "chan<- string")
	testType(t, 13, typ.(*ChanType).Elem(), "string")

	// make sure tag strings are not part of element type
	typ = Typeof(struct {
		d []uint32 "TAG"
	}{}).(*StructType).Field(0).Type
	testType(t, 14, typ, "[]uint32")
}

func TestInterfaceGet(t *testing.T) {
	var inter struct {
		e interface{}
	}
	inter.e = 123.456
	v1 := NewValue(&inter)
	v2 := v1.(*PtrValue).Elem().(*StructValue).Field(0)
	assert(t, v2.Type().String(), "interface { }")
	i2 := v2.(*InterfaceValue).Interface()
	v3 := NewValue(i2)
	assert(t, v3.Type().String(), "float")
}

func TestInterfaceValue(t *testing.T) {
	var inter struct {
		e interface{}
	}
	inter.e = 123.456
	v1 := NewValue(&inter)
	v2 := v1.(*PtrValue).Elem().(*StructValue).Field(0)
	assert(t, v2.Type().String(), "interface { }")
	v3 := v2.(*InterfaceValue).Elem()
	assert(t, v3.Type().String(), "float")

	i3 := v2.Interface()
	if _, ok := i3.(float); !ok {
		t.Error("v2.Interface() did not return float, got ", Typeof(i3))
	}
}

func TestFunctionValue(t *testing.T) {
	v := NewValue(func() {})
	if v.Interface() != v.Interface() {
		t.Fatalf("TestFunction != itself")
	}
	assert(t, v.Type().String(), "func()")
}

func TestCopyArray(t *testing.T) {
	a := []int{1, 2, 3, 4, 10, 9, 8, 7}
	b := []int{11, 22, 33, 44, 1010, 99, 88, 77, 66, 55, 44}
	c := []int{11, 22, 33, 44, 1010, 99, 88, 77, 66, 55, 44}
	va := NewValue(&a)
	vb := NewValue(&b)
	for i := 0; i < len(b); i++ {
		if b[i] != c[i] {
			t.Fatalf("b != c before test")
		}
	}
	aa := va.(*PtrValue).Elem().(*SliceValue)
	ab := vb.(*PtrValue).Elem().(*SliceValue)
	for tocopy := 1; tocopy <= 7; tocopy++ {
		aa.SetLen(tocopy)
		ArrayCopy(ab, aa)
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
	DeepEqualTest{1, 1, true},
	DeepEqualTest{int32(1), int32(1), true},
	DeepEqualTest{0.5, 0.5, true},
	DeepEqualTest{float32(0.5), float32(0.5), true},
	DeepEqualTest{"hello", "hello", true},
	DeepEqualTest{make([]int, 10), make([]int, 10), true},
	DeepEqualTest{&[3]int{1, 2, 3}, &[3]int{1, 2, 3}, true},
	DeepEqualTest{Basic{1, 0.5}, Basic{1, 0.5}, true},
	DeepEqualTest{os.Error(nil), os.Error(nil), true},
	DeepEqualTest{map[int]string{1: "one", 2: "two"}, map[int]string{2: "two", 1: "one"}, true},

	// Inequalities
	DeepEqualTest{1, 2, false},
	DeepEqualTest{int32(1), int32(2), false},
	DeepEqualTest{0.5, 0.6, false},
	DeepEqualTest{float32(0.5), float32(0.6), false},
	DeepEqualTest{"hello", "hey", false},
	DeepEqualTest{make([]int, 10), make([]int, 11), false},
	DeepEqualTest{&[3]int{1, 2, 3}, &[3]int{1, 2, 4}, false},
	DeepEqualTest{Basic{1, 0.5}, Basic{1, 0.6}, false},
	DeepEqualTest{Basic{1, 0}, Basic{2, 0}, false},
	DeepEqualTest{map[int]string{1: "one", 3: "two"}, map[int]string{2: "two", 1: "one"}, false},
	DeepEqualTest{map[int]string{1: "one", 2: "txo"}, map[int]string{2: "two", 1: "one"}, false},
	DeepEqualTest{map[int]string{1: "one"}, map[int]string{2: "two", 1: "one"}, false},
	DeepEqualTest{map[int]string{2: "two", 1: "one"}, map[int]string{1: "one"}, false},
	DeepEqualTest{nil, 1, false},
	DeepEqualTest{1, nil, false},

	// Mismatched types
	DeepEqualTest{1, 1.0, false},
	DeepEqualTest{int32(1), int64(1), false},
	DeepEqualTest{0.5, "hello", false},
	DeepEqualTest{[]int{1, 2, 3}, [3]int{1, 2, 3}, false},
	DeepEqualTest{&[3]interface{}{1, 2, 4}, &[3]interface{}{1, 2, "s"}, false},
	DeepEqualTest{Basic{1, 0.5}, NotBasic{1, 0.5}, false},
	DeepEqualTest{map[uint]string{1: "one", 2: "two"}, map[int]string{2: "two", 1: "one"}, false},
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
		if v == nil {
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

type Complex struct {
	a int
	b [3]*Complex
	c *string
	d map[float]float
}

func TestDeepEqualComplexStruct(t *testing.T) {
	m := make(map[float]float)
	stra, strb := "hello", "hello"
	a, b := new(Complex), new(Complex)
	*a = Complex{5, [3]*Complex{a, b, a}, &stra, m}
	*b = Complex{5, [3]*Complex{b, a, a}, &strb, m}
	if !DeepEqual(a, b) {
		t.Error("DeepEqual(complex same) = false, want true")
	}
}

func TestDeepEqualComplexStructInequality(t *testing.T) {
	m := make(map[float]float)
	stra, strb := "hello", "helloo" // Difference is here
	a, b := new(Complex), new(Complex)
	*a = Complex{5, [3]*Complex{a, b, a}, &stra, m}
	*b = Complex{5, [3]*Complex{b, a, a}, &strb, m}
	if DeepEqual(a, b) {
		t.Error("DeepEqual(complex different) = true, want false")
	}
}


func check2ndField(x interface{}, offs uintptr, t *testing.T) {
	s := NewValue(x).(*StructValue)
	f := s.Type().(*StructType).Field(1)
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

type IsNiller interface {
	IsNil() bool
}

func Nil(a interface{}, t *testing.T) {
	n := NewValue(a).(*StructValue).Field(0).(IsNiller)
	if !n.IsNil() {
		t.Errorf("%v should be nil", a)
	}
}

func NotNil(a interface{}, t *testing.T) {
	n := NewValue(a).(*StructValue).Field(0).(IsNiller)
	if n.IsNil() {
		t.Errorf("value of type %v should not be nil", NewValue(a).Type().String())
	}
}

func TestIsNil(t *testing.T) {
	// These do not implement IsNil
	doNotNil := []interface{}{int(0), float32(0), struct{ a int }{}}
	for _, ts := range doNotNil {
		ty := Typeof(ts)
		v := MakeZero(ty)
		if _, ok := v.(IsNiller); ok {
			t.Errorf("%s is nilable; should not be", ts)
		}
	}

	// These do implement IsNil.
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
		ty := Typeof(ts).(*StructType).Field(0).Type
		v := MakeZero(ty)
		if _, ok := v.(IsNiller); !ok {
			t.Errorf("%s %T is not nilable; should be", ts, v)
		}
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
	v := Indirect(NewValue(&s)).(*StructValue).Field(0).Interface()
	if v != s.w.(interface{}) {
		t.Error("Interface() on interface: ", v, s.w)
	}
}

func TestInterfaceEditing(t *testing.T) {
	// strings are bigger than one word,
	// so the interface conversion allocates
	// memory to hold a string and puts that
	// pointer in the interface.
	var i interface{} = "hello"

	// if i pass the interface value by value
	// to NewValue, i should get a fresh copy
	// of the value.
	v := NewValue(i)

	// and setting that copy to "bye" should
	// not change the value stored in i.
	v.(*StringValue).Set("bye")
	if i.(string) != "hello" {
		t.Errorf(`Set("bye") changed i to %s`, i.(string))
	}

	// the same should be true of smaller items.
	i = 123
	v = NewValue(i)
	v.(*IntValue).Set(234)
	if i.(int) != 123 {
		t.Errorf("Set(234) changed i to %d", i.(int))
	}
}

func TestNilPtrValueSub(t *testing.T) {
	var pi *int
	if pv := NewValue(pi).(*PtrValue); pv.Elem() != nil {
		t.Error("NewValue((*int)(nil)).(*PtrValue).Elem() != nil")
	}
}

func TestMap(t *testing.T) {
	m := map[string]int{"a": 1, "b": 2}
	mv := NewValue(m).(*MapValue)
	if n := mv.Len(); n != len(m) {
		t.Errorf("Len = %d, want %d", n, len(m))
	}
	keys := mv.Keys()
	i := 0
	newmap := MakeMap(mv.Type().(*MapType))
	for k, v := range m {
		// Check that returned Keys match keys in range.
		// These aren't required to be in the same order,
		// but they are in this implementation, which makes
		// the test easier.
		if i >= len(keys) {
			t.Errorf("Missing key #%d %q", i, k)
		} else if kv := keys[i].(*StringValue); kv.Get() != k {
			t.Errorf("Keys[%d] = %q, want %q", i, kv.Get(), k)
		}
		i++

		// Check that value lookup is correct.
		vv := mv.Elem(NewValue(k))
		if vi := vv.(*IntValue).Get(); vi != v {
			t.Errorf("Key %q: have value %d, want %d", vi, v)
		}

		// Copy into new map.
		newmap.SetElem(NewValue(k), NewValue(v))
	}
	vv := mv.Elem(NewValue("not-present"))
	if vv != nil {
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

	newmap.SetElem(NewValue("a"), nil)
	v, ok := newm["a"]
	if ok {
		t.Errorf("newm[\"a\"] = %d after delete", v)
	}
}

func TestChan(t *testing.T) {
	for loop := 0; loop < 2; loop++ {
		var c chan int
		var cv *ChanValue

		// check both ways to allocate channels
		switch loop {
		case 1:
			c = make(chan int, 1)
			cv = NewValue(c).(*ChanValue)
		case 0:
			cv = MakeChan(Typeof(c).(*ChanType), 1)
			c = cv.Interface().(chan int)
		}

		// Send
		cv.Send(NewValue(2))
		if i := <-c; i != 2 {
			t.Errorf("reflect Send 2, native recv %d", i)
		}

		// Recv
		c <- 3
		if i := cv.Recv().(*IntValue).Get(); i != 3 {
			t.Errorf("native send 3, reflect Recv %d", i)
		}

		// TryRecv fail
		val := cv.TryRecv()
		if val != nil {
			t.Errorf("TryRecv on empty chan: %s", valueToString(val))
		}

		// TryRecv success
		c <- 4
		val = cv.TryRecv()
		if val == nil {
			t.Errorf("TryRecv on ready chan got nil")
		} else if i := val.(*IntValue).Get(); i != 4 {
			t.Errorf("native send 4, TryRecv %d", i)
		}

		// TrySend fail
		c <- 100
		ok := cv.TrySend(NewValue(5))
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
		if cv.Closed() {
			t.Errorf("closed too soon - 1")
		}
		if i := cv.Recv().(*IntValue).Get(); i != 123 {
			t.Errorf("send 123 then close; Recv %d", i)
		}
		if cv.Closed() {
			t.Errorf("closed too soon - 2")
		}
		if i := cv.Recv().(*IntValue).Get(); i != 0 {
			t.Errorf("after close Recv %d", i)
		}
		if !cv.Closed() {
			t.Errorf("not closed")
		}
	}

	// check creation of unbuffered channel
	var c chan int
	cv := MakeChan(Typeof(c).(*ChanType), 0)
	c = cv.Interface().(chan int)
	if cv.TrySend(NewValue(7)) {
		t.Errorf("TrySend on sync chan succeeded")
	}
	if cv.TryRecv() != nil {
		t.Errorf("TryRecv on sync chan succeeded")
	}

	// len/cap
	cv = MakeChan(Typeof(c).(*ChanType), 10)
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
	ret := NewValue(dummy).(*FuncValue).Call([]Value{NewValue(byte(10)), NewValue(20), NewValue(byte(30))})
	if len(ret) != 3 {
		t.Fatalf("Call returned %d values, want 3", len(ret))
	}

	i := ret[0].(*Uint8Value).Get()
	j := ret[1].(*IntValue).Get()
	k := ret[2].(*Uint8Value).Get()
	if i != 10 || j != 20 || k != 30 {
		t.Errorf("Call returned %d, %d, %d; want 10, 20, 30", i, j, k)
	}
}

type Point struct {
	x, y int
}

func (p Point) Dist(scale int) int { return p.x*p.x*scale + p.y*p.y*scale }

func TestMethod(t *testing.T) {
	// Non-curried method of type.
	p := Point{3, 4}
	i := Typeof(p).Method(0).Func.Call([]Value{NewValue(p), NewValue(10)})[0].(*IntValue).Get()
	if i != 250 {
		t.Errorf("Type Method returned %d; want 250", i)
	}

	// Curried method of value.
	i = NewValue(p).Method(0).Call([]Value{NewValue(10)})[0].(*IntValue).Get()
	if i != 250 {
		t.Errorf("Value Method returned %d; want 250", i)
	}

	// Curried method of interface value.
	// Have to wrap interface value in a struct to get at it.
	// Passing it to NewValue directly would
	// access the underlying Point, not the interface.
	var s = struct {
		x interface {
			Dist(int) int
		}
	}{p}
	pv := NewValue(s).(*StructValue).Field(0)
	i = pv.Method(0).Call([]Value{NewValue(10)})[0].(*IntValue).Get()
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
	sv := NewValue(&s).(*PtrValue).Elem().(*StructValue)
	sv.Field(0).(*InterfaceValue).Set(NewValue(p))
	if q := s.I.(*Point); q != p {
		t.Errorf("i: have %p want %p", q, p)
	}

	pv := sv.Field(1).(*InterfaceValue)
	pv.Set(NewValue(p))
	if q := s.P.(*Point); q != p {
		t.Errorf("i: have %p want %p", q, p)
	}

	i := pv.Method(0).Call([]Value{NewValue(10)})[0].(*IntValue).Get()
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
	type1 := Typeof(t1).(*StructType)
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
	FTest{struct{}{}, "", nil, 0},
	FTest{struct{}{}, "foo", nil, 0},
	FTest{S0{a: 'a'}, "a", []int{0}, 'a'},
	FTest{S0{}, "d", nil, 0},
	FTest{S1{S0: S0{a: 'a'}}, "a", []int{1, 0}, 'a'},
	FTest{S1{b: 'b'}, "b", []int{0}, 'b'},
	FTest{S1{}, "S0", []int{1}, 0},
	FTest{S1{S0: S0{c: 'c'}}, "c", []int{1, 2}, 'c'},
	FTest{S2{a: 'a'}, "a", []int{0}, 'a'},
	FTest{S2{}, "S1", []int{1}, 0},
	FTest{S2{S1: &S1{b: 'b'}}, "b", []int{1, 0}, 'b'},
	FTest{S2{S1: &S1{S0: S0{c: 'c'}}}, "c", []int{1, 1, 2}, 'c'},
	FTest{S2{}, "d", nil, 0},
	FTest{S3{}, "S1", nil, 0},
	FTest{S3{S2: S2{a: 'a'}}, "a", []int{1, 0}, 'a'},
	FTest{S3{}, "b", nil, 0},
	FTest{S3{d: 'd'}, "d", []int{2}, 0},
	FTest{S3{e: 'e'}, "e", []int{3}, 'e'},
	FTest{S4{a: 'a'}, "a", []int{1}, 'a'},
	FTest{S4{}, "b", nil, 0},
}

func TestFieldByIndex(t *testing.T) {
	for _, test := range fieldTests {
		s := Typeof(test.s).(*StructType)
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
			v := NewValue(test.s).(*StructValue).FieldByIndex(test.index)
			if v != nil {
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
		s := Typeof(test.s).(*StructType)
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
			v := NewValue(test.s).(*StructValue).FieldByName(test.name)
			if v != nil {
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
