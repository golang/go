// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflectlite_test

import (
	"encoding/base64"
	"fmt"
	"internal/abi"
	. "internal/reflectlite"
	"math"
	"reflect"
	"runtime"
	"testing"
	"unsafe"
)

func ToValue(v Value) reflect.Value {
	return reflect.ValueOf(ToInterface(v))
}

func TypeString(t Type) string {
	return fmt.Sprintf("%T", ToInterface(Zero(t)))
}

type integer int
type T struct {
	a int
	b float64
	c string
	d *int
}

type pair struct {
	i any
	s string
}

func assert(t *testing.T, s, want string) {
	t.Helper()
	if s != want {
		t.Errorf("have %#q want %#q", s, want)
	}
}

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
	{struct{ x (**integer) }{}, "**reflectlite_test.integer"},
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
		"struct { c func(chan *reflectlite_test.integer, *int8) }",
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
	// {struct {
	// 	x (interface {
	// 		a(func(func(int) int) func(func(int)) int)
	// 		b()
	// 	})
	// }{},
	// 	"interface { reflectlite_test.a(func(func(int) int) func(func(int)) int); reflectlite_test.b() }",
	// },
	{struct {
		x struct {
			int32
			int64
		}
	}{},
		"struct { int32; int64 }",
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
	{new(**integer), "**reflectlite_test.integer(0)"},
	{new(map[string]int32), "map[string]int32{<can't iterate on maps>}"},
	{new(chan<- string), "chan<- string"},
	{new(func(a int8, b int32)), "func(int8, int32)(arg)"},
	{new(struct {
		c chan *int32
		d float32
	}),
		"struct { c chan *int32; d float32 }{chan *int32, 0}",
	},
	{new(struct{ c func(chan *integer, *int8) }),
		"struct { c func(chan *reflectlite_test.integer, *int8) }{func(chan *reflectlite_test.integer, *int8)(arg)}",
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
	s := TypeString(typ)
	if s != want {
		t.Errorf("#%d: have %#q, want %#q", i, s, want)
	}
}

func testReflectType(t *testing.T, i int, typ Type, want string) {
	s := TypeString(typ)
	if s != want {
		t.Errorf("#%d: have %#q, want %#q", i, s, want)
	}
}

func TestTypes(t *testing.T) {
	for i, tt := range typeTests {
		testReflectType(t, i, Field(ValueOf(tt.i), 0).Type(), tt.s)
	}
}

func TestSetValue(t *testing.T) {
	for i, tt := range valueTests {
		v := ValueOf(tt.i).Elem()
		switch v.Kind() {
		case abi.Int:
			v.Set(ValueOf(int(132)))
		case abi.Int8:
			v.Set(ValueOf(int8(8)))
		case abi.Int16:
			v.Set(ValueOf(int16(16)))
		case abi.Int32:
			v.Set(ValueOf(int32(32)))
		case abi.Int64:
			v.Set(ValueOf(int64(64)))
		case abi.Uint:
			v.Set(ValueOf(uint(132)))
		case abi.Uint8:
			v.Set(ValueOf(uint8(8)))
		case abi.Uint16:
			v.Set(ValueOf(uint16(16)))
		case abi.Uint32:
			v.Set(ValueOf(uint32(32)))
		case abi.Uint64:
			v.Set(ValueOf(uint64(64)))
		case abi.Float32:
			v.Set(ValueOf(float32(256.25)))
		case abi.Float64:
			v.Set(ValueOf(512.125))
		case abi.Complex64:
			v.Set(ValueOf(complex64(532.125 + 10i)))
		case abi.Complex128:
			v.Set(ValueOf(complex128(564.25 + 1i)))
		case abi.String:
			v.Set(ValueOf("stringy cheese"))
		case abi.Bool:
			v.Set(ValueOf(true))
		}
		s := valueToString(v)
		if s != tt.s {
			t.Errorf("#%d: have %#q, want %#q", i, s, tt.s)
		}
	}
}

func TestCanSetField(t *testing.T) {
	type embed struct{ x, X int }
	type Embed struct{ x, X int }
	type S1 struct {
		embed
		x, X int
	}
	type S2 struct {
		*embed
		x, X int
	}
	type S3 struct {
		Embed
		x, X int
	}
	type S4 struct {
		*Embed
		x, X int
	}

	type testCase struct {
		index  []int
		canSet bool
	}
	tests := []struct {
		val   Value
		cases []testCase
	}{{
		val: ValueOf(&S1{}),
		cases: []testCase{
			{[]int{0}, false},
			{[]int{0, 0}, false},
			{[]int{0, 1}, true},
			{[]int{1}, false},
			{[]int{2}, true},
		},
	}, {
		val: ValueOf(&S2{embed: &embed{}}),
		cases: []testCase{
			{[]int{0}, false},
			{[]int{0, 0}, false},
			{[]int{0, 1}, true},
			{[]int{1}, false},
			{[]int{2}, true},
		},
	}, {
		val: ValueOf(&S3{}),
		cases: []testCase{
			{[]int{0}, true},
			{[]int{0, 0}, false},
			{[]int{0, 1}, true},
			{[]int{1}, false},
			{[]int{2}, true},
		},
	}, {
		val: ValueOf(&S4{Embed: &Embed{}}),
		cases: []testCase{
			{[]int{0}, true},
			{[]int{0, 0}, false},
			{[]int{0, 1}, true},
			{[]int{1}, false},
			{[]int{2}, true},
		},
	}}

	for _, tt := range tests {
		t.Run(tt.val.Type().Name(), func { t ->
			for _, tc := range tt.cases {
				f := tt.val
				for _, i := range tc.index {
					if f.Kind() == Ptr {
						f = f.Elem()
					}
					f = Field(f, i)
				}
				if got := f.CanSet(); got != tc.canSet {
					t.Errorf("CanSet() = %v, want %v", got, tc.canSet)
				}
			}
		})
	}
}

var _i = 7

var valueToStringTests = []pair{
	{123, "123"},
	{123.5, "123.5"},
	{byte(123), "123"},
	{"abc", "abc"},
	{T{123, 456.75, "hello", &_i}, "reflectlite_test.T{123, 456.75, hello, *int(&7)}"},
	{new(chan *T), "*chan *reflectlite_test.T(&chan *reflectlite_test.T)"},
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
}

func TestInterfaceValue(t *testing.T) {
	var inter struct {
		E any
	}
	inter.E = 123.456
	v1 := ValueOf(&inter)
	v2 := Field(v1.Elem(), 0)
	// assert(t, TypeString(v2.Type()), "interface {}")
	v3 := v2.Elem()
	assert(t, TypeString(v3.Type()), "float64")

	i3 := ToInterface(v2)
	if _, ok := i3.(float64); !ok {
		t.Error("v2.Interface() did not return float64, got ", TypeOf(i3))
	}
}

func TestFunctionValue(t *testing.T) {
	var x any = func() {}
	v := ValueOf(x)
	if fmt.Sprint(ToInterface(v)) != fmt.Sprint(x) {
		t.Fatalf("TestFunction returned wrong pointer")
	}
	assert(t, TypeString(v.Type()), "func()")
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

func TestBigUnnamedStruct(t *testing.T) {
	b := struct{ a, b, c, d int64 }{1, 2, 3, 4}
	v := ValueOf(b)
	b1 := ToInterface(v).(struct {
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
	b1 := ToInterface(v).(big)
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
	a, b any
	eq   bool
}

// Simple functions for DeepEqual tests.
var (
	fn1 func()             // nil.
	fn2 func()             // nil.
	fn3 = func() { fn1() } // Not nil.
)

type self struct{}

type Loop *Loop
type Loopy any

var loop1, loop2 Loop
var loopy1, loopy2 Loopy

func init() {
	loop1 = &loop2
	loop2 = &loop1

	loopy1 = &loopy2
	loopy2 = &loopy1
}

var typeOfTests = []DeepEqualTest{
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
	{math.NaN(), math.NaN(), false},
	{&[1]float64{math.NaN()}, &[1]float64{math.NaN()}, false},
	{&[1]float64{math.NaN()}, self{}, true},
	{[]float64{math.NaN()}, []float64{math.NaN()}, false},
	{[]float64{math.NaN()}, self{}, true},
	{map[float64]float64{math.NaN(): 1}, map[float64]float64{1: 2}, false},
	{map[float64]float64{math.NaN(): 1}, self{}, true},

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
	{&[3]any{1, 2, 4}, &[3]any{1, 2, "s"}, false},
	{Basic{1, 0.5}, NotBasic{1, 0.5}, false},
	{map[uint]string{1: "one", 2: "two"}, map[int]string{2: "two", 1: "one"}, false},

	// Possible loops.
	{&loop1, &loop1, true},
	{&loop1, &loop2, true},
	{&loopy1, &loopy1, true},
	{&loopy1, &loopy2, true},
}

func TestTypeOf(t *testing.T) {
	// Special case for nil
	if typ := TypeOf(nil); typ != nil {
		t.Errorf("expected nil type for nil value; got %v", typ)
	}
	for _, test := range typeOfTests {
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

func Nil(a any, t *testing.T) {
	n := Field(ValueOf(a), 0)
	if !n.IsNil() {
		t.Errorf("%v should be nil", a)
	}
}

func NotNil(a any, t *testing.T) {
	n := Field(ValueOf(a), 0)
	if n.IsNil() {
		t.Errorf("value of type %v should not be nil", TypeString(ValueOf(a).Type()))
	}
}

func TestIsNil(t *testing.T) {
	// These implement IsNil.
	// Wrap in extra struct to hide interface type.
	doNil := []any{
		struct{ x *int }{},
		struct{ x any }{},
		struct{ x map[string]int }{},
		struct{ x func() bool }{},
		struct{ x chan int }{},
		struct{ x []string }{},
		struct{ x unsafe.Pointer }{},
	}
	for _, ts := range doNil {
		ty := TField(TypeOf(ts), 0)
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
		x any
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

// Indirect returns the value that v points to.
// If v is a nil pointer, Indirect returns a zero Value.
// If v is not a pointer, Indirect returns v.
func Indirect(v Value) Value {
	if v.Kind() != Ptr {
		return v
	}
	return v.Elem()
}

func TestNilPtrValueSub(t *testing.T) {
	var pi *int
	if pv := ValueOf(pi); pv.Elem().IsValid() {
		t.Error("ValueOf((*int)(nil)).Elem().IsValid()")
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
func (p Point) NoArgs() {
	// Exercise no-argument/no-result paths.
}

// This will be index 4.
func (p Point) TotalDist(points ...Point) int {
	tot := 0
	for _, q := range points {
		dx := q.x - p.x
		dy := q.y - p.y
		tot += dx*dx + dy*dy // Should call Sqrt, but it's just a test.

	}
	return tot
}

type D1 struct {
	d int
}
type D2 struct {
	d int
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
		{TypeOf((*any)(nil)).Elem(), ""},
		{TypeOf((*byte)(nil)), ""},
		{TypeOf((*rune)(nil)), ""},
		{TypeOf((*int64)(nil)), ""},
		{TypeOf(map[string]int{}), ""},
		{TypeOf((*error)(nil)).Elem(), ""},
		{TypeOf((*Point)(nil)), ""},
		{TypeOf((*Point)(nil)).Elem(), "internal/reflectlite_test"},
	}
	for _, test := range tests {
		if path := test.t.PkgPath(); path != test.path {
			t.Errorf("%v.PkgPath() = %q, want %q", test.t, path, test.path)
		}
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
	noAlloc(t, 100, func { j ->
		var i any
		var v Value

		i = []int{j, j, j}
		v = ValueOf(i)
		if v.Len() != 3 {
			panic("wrong length")
		}
	})
	noAlloc(t, 100, func { j ->
		var i any
		var v Value

		i = func { j -> j }
		v = ValueOf(i)
		if ToInterface(v).(func(int) int)(j) != j {
			panic("wrong result")
		}
	})
}

func TestSetPanic(t *testing.T) {
	ok := func(f func()) { f() }
	bad := shouldPanic
	clear := func(v Value) { v.Set(Zero(v.Type())) }

	type t0 struct {
		W int
	}

	type t1 struct {
		Y int
		t0
	}

	type T2 struct {
		Z       int
		namedT0 t0
	}

	type T struct {
		X int
		t1
		T2
		NamedT1 t1
		NamedT2 T2
		namedT1 t1
		namedT2 T2
	}

	// not addressable
	v := ValueOf(T{})
	bad(func() { clear(Field(v, 0)) })                     // .X
	bad(func() { clear(Field(v, 1)) })                     // .t1
	bad(func() { clear(Field(Field(v, 1), 0)) })           // .t1.Y
	bad(func() { clear(Field(Field(v, 1), 1)) })           // .t1.t0
	bad(func() { clear(Field(Field(Field(v, 1), 1), 0)) }) // .t1.t0.W
	bad(func() { clear(Field(v, 2)) })                     // .T2
	bad(func() { clear(Field(Field(v, 2), 0)) })           // .T2.Z
	bad(func() { clear(Field(Field(v, 2), 1)) })           // .T2.namedT0
	bad(func() { clear(Field(Field(Field(v, 2), 1), 0)) }) // .T2.namedT0.W
	bad(func() { clear(Field(v, 3)) })                     // .NamedT1
	bad(func() { clear(Field(Field(v, 3), 0)) })           // .NamedT1.Y
	bad(func() { clear(Field(Field(v, 3), 1)) })           // .NamedT1.t0
	bad(func() { clear(Field(Field(Field(v, 3), 1), 0)) }) // .NamedT1.t0.W
	bad(func() { clear(Field(v, 4)) })                     // .NamedT2
	bad(func() { clear(Field(Field(v, 4), 0)) })           // .NamedT2.Z
	bad(func() { clear(Field(Field(v, 4), 1)) })           // .NamedT2.namedT0
	bad(func() { clear(Field(Field(Field(v, 4), 1), 0)) }) // .NamedT2.namedT0.W
	bad(func() { clear(Field(v, 5)) })                     // .namedT1
	bad(func() { clear(Field(Field(v, 5), 0)) })           // .namedT1.Y
	bad(func() { clear(Field(Field(v, 5), 1)) })           // .namedT1.t0
	bad(func() { clear(Field(Field(Field(v, 5), 1), 0)) }) // .namedT1.t0.W
	bad(func() { clear(Field(v, 6)) })                     // .namedT2
	bad(func() { clear(Field(Field(v, 6), 0)) })           // .namedT2.Z
	bad(func() { clear(Field(Field(v, 6), 1)) })           // .namedT2.namedT0
	bad(func() { clear(Field(Field(Field(v, 6), 1), 0)) }) // .namedT2.namedT0.W

	// addressable
	v = ValueOf(&T{}).Elem()
	ok(func() { clear(Field(v, 0)) })                      // .X
	bad(func() { clear(Field(v, 1)) })                     // .t1
	ok(func() { clear(Field(Field(v, 1), 0)) })            // .t1.Y
	bad(func() { clear(Field(Field(v, 1), 1)) })           // .t1.t0
	ok(func() { clear(Field(Field(Field(v, 1), 1), 0)) })  // .t1.t0.W
	ok(func() { clear(Field(v, 2)) })                      // .T2
	ok(func() { clear(Field(Field(v, 2), 0)) })            // .T2.Z
	bad(func() { clear(Field(Field(v, 2), 1)) })           // .T2.namedT0
	bad(func() { clear(Field(Field(Field(v, 2), 1), 0)) }) // .T2.namedT0.W
	ok(func() { clear(Field(v, 3)) })                      // .NamedT1
	ok(func() { clear(Field(Field(v, 3), 0)) })            // .NamedT1.Y
	bad(func() { clear(Field(Field(v, 3), 1)) })           // .NamedT1.t0
	ok(func() { clear(Field(Field(Field(v, 3), 1), 0)) })  // .NamedT1.t0.W
	ok(func() { clear(Field(v, 4)) })                      // .NamedT2
	ok(func() { clear(Field(Field(v, 4), 0)) })            // .NamedT2.Z
	bad(func() { clear(Field(Field(v, 4), 1)) })           // .NamedT2.namedT0
	bad(func() { clear(Field(Field(Field(v, 4), 1), 0)) }) // .NamedT2.namedT0.W
	bad(func() { clear(Field(v, 5)) })                     // .namedT1
	bad(func() { clear(Field(Field(v, 5), 0)) })           // .namedT1.Y
	bad(func() { clear(Field(Field(v, 5), 1)) })           // .namedT1.t0
	bad(func() { clear(Field(Field(Field(v, 5), 1), 0)) }) // .namedT1.t0.W
	bad(func() { clear(Field(v, 6)) })                     // .namedT2
	bad(func() { clear(Field(Field(v, 6), 0)) })           // .namedT2.Z
	bad(func() { clear(Field(Field(v, 6), 1)) })           // .namedT2.namedT0
	bad(func() { clear(Field(Field(Field(v, 6), 1), 0)) }) // .namedT2.namedT0.W
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("did not panic")
		}
	}()
	f()
}

type S struct {
	i1 int64
	i2 int64
}

func TestBigZero(t *testing.T) {
	const size = 1 << 10
	var v [size]byte
	z := ToInterface(Zero(ValueOf(v).Type())).([size]byte)
	for i := 0; i < size; i++ {
		if z[i] != 0 {
			t.Fatalf("Zero object not all zero, index %d", i)
		}
	}
}

func TestInvalid(t *testing.T) {
	// Used to have inconsistency between IsValid() and Kind() != Invalid.
	type T struct{ v any }

	v := Field(ValueOf(T{}), 0)
	if v.IsValid() != true || v.Kind() != Interface {
		t.Errorf("field: IsValid=%v, Kind=%v, want true, Interface", v.IsValid(), v.Kind())
	}
	v = v.Elem()
	if v.IsValid() != false || v.Kind() != abi.Invalid {
		t.Errorf("field elem: IsValid=%v, Kind=%v, want false, Invalid", v.IsValid(), v.Kind())
	}
}

type TheNameOfThisTypeIsExactly255BytesLongSoWhenTheCompilerPrependsTheReflectTestPackageNameAndExtraStarTheLinkerRuntimeAndReflectPackagesWillHaveToCorrectlyDecodeTheSecondLengthByte0123456789_0123456789_0123456789_0123456789_0123456789_012345678 int

type nameTest struct {
	v    any
	want string
}

type A struct{}
type B[T any] struct{}

var nameTests = []nameTest{
	{(*int32)(nil), "int32"},
	{(*D1)(nil), "D1"},
	{(*[]D1)(nil), ""},
	{(*chan D1)(nil), ""},
	{(*func() D1)(nil), ""},
	{(*<-chan D1)(nil), ""},
	{(*chan<- D1)(nil), ""},
	{(*any)(nil), ""},
	{(*interface {
		F()
	})(nil), ""},
	{(*TheNameOfThisTypeIsExactly255BytesLongSoWhenTheCompilerPrependsTheReflectTestPackageNameAndExtraStarTheLinkerRuntimeAndReflectPackagesWillHaveToCorrectlyDecodeTheSecondLengthByte0123456789_0123456789_0123456789_0123456789_0123456789_012345678)(nil), "TheNameOfThisTypeIsExactly255BytesLongSoWhenTheCompilerPrependsTheReflectTestPackageNameAndExtraStarTheLinkerRuntimeAndReflectPackagesWillHaveToCorrectlyDecodeTheSecondLengthByte0123456789_0123456789_0123456789_0123456789_0123456789_012345678"},
	{(*B[A])(nil), "B[internal/reflectlite_test.A]"},
	{(*B[B[A]])(nil), "B[internal/reflectlite_test.B[internal/reflectlite_test.A]]"},
}

func TestNames(t *testing.T) {
	for _, test := range nameTests {
		typ := TypeOf(test.v).Elem()
		if got := typ.Name(); got != test.want {
			t.Errorf("%v Name()=%q, want %q", typ, got, test.want)
		}
	}
}

// TestUnaddressableField tests that the reflect package will not allow
// a type from another package to be used as a named type with an
// unexported field.
//
// This ensures that unexported fields cannot be modified by other packages.
func TestUnaddressableField(t *testing.T) {
	var b Buffer // type defined in reflect, a different package
	var localBuffer struct {
		buf []byte
	}
	lv := ValueOf(&localBuffer).Elem()
	rv := ValueOf(b)
	shouldPanic(func() {
		lv.Set(rv)
	})
}

type Tint int

type Tint2 = Tint

type Talias1 struct {
	byte
	uint8
	int
	int32
	rune
}

type Talias2 struct {
	Tint
	Tint2
}

func TestAliasNames(t *testing.T) {
	t1 := Talias1{byte: 1, uint8: 2, int: 3, int32: 4, rune: 5}
	out := fmt.Sprintf("%#v", t1)
	want := "reflectlite_test.Talias1{byte:0x1, uint8:0x2, int:3, int32:4, rune:5}"
	if out != want {
		t.Errorf("Talias1 print:\nhave: %s\nwant: %s", out, want)
	}

	t2 := Talias2{Tint: 1, Tint2: 2}
	out = fmt.Sprintf("%#v", t2)
	want = "reflectlite_test.Talias2{Tint:1, Tint2:2}"
	if out != want {
		t.Errorf("Talias2 print:\nhave: %s\nwant: %s", out, want)
	}
}
