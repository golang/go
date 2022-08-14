// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"strings"
	"testing"

	. "cmd/compile/internal/types2"
)

var builtinCalls = []struct {
	name, src, sig string
}{
	{"append", `var s []int; _ = append(s)`, `func([]int, ...int) []int`},
	{"append", `var s []int; _ = append(s, 0)`, `func([]int, ...int) []int`},
	{"append", `var s []int; _ = (append)(s, 0)`, `func([]int, ...int) []int`},
	{"append", `var s []byte; _ = ((append))(s, 0)`, `func([]byte, ...byte) []byte`},
	{"append", `var s []byte; _ = append(s, "foo"...)`, `func([]byte, string...) []byte`},
	{"append", `type T []byte; var s T; var str string; _ = append(s, str...)`, `func(p.T, string...) p.T`},
	{"append", `type T []byte; type U string; var s T; var str U; _ = append(s, str...)`, `func(p.T, p.U...) p.T`},

	{"cap", `var s [10]int; _ = cap(s)`, `invalid type`},  // constant
	{"cap", `var s [10]int; _ = cap(&s)`, `invalid type`}, // constant
	{"cap", `var s []int64; _ = cap(s)`, `func([]int64) int`},
	{"cap", `var c chan<-bool; _ = cap(c)`, `func(chan<- bool) int`},
	{"cap", `type S []byte; var s S; _ = cap(s)`, `func(p.S) int`},
	{"cap", `var s P; _ = cap(s)`, `func(P) int`},

	{"len", `_ = len("foo")`, `invalid type`}, // constant
	{"len", `var s string; _ = len(s)`, `func(string) int`},
	{"len", `var s [10]int; _ = len(s)`, `invalid type`},  // constant
	{"len", `var s [10]int; _ = len(&s)`, `invalid type`}, // constant
	{"len", `var s []int64; _ = len(s)`, `func([]int64) int`},
	{"len", `var c chan<-bool; _ = len(c)`, `func(chan<- bool) int`},
	{"len", `var m map[string]float32; _ = len(m)`, `func(map[string]float32) int`},
	{"len", `type S []byte; var s S; _ = len(s)`, `func(p.S) int`},
	{"len", `var s P; _ = len(s)`, `func(P) int`},

	{"close", `var c chan int; close(c)`, `func(chan int)`},
	{"close", `var c chan<- chan string; close(c)`, `func(chan<- chan string)`},

	{"complex", `_ = complex(1, 0)`, `invalid type`}, // constant
	{"complex", `var re float32; _ = complex(re, 1.0)`, `func(float32, float32) complex64`},
	{"complex", `var im float64; _ = complex(1, im)`, `func(float64, float64) complex128`},
	{"complex", `type F32 float32; var re, im F32; _ = complex(re, im)`, `func(p.F32, p.F32) complex64`},
	{"complex", `type F64 float64; var re, im F64; _ = complex(re, im)`, `func(p.F64, p.F64) complex128`},

	{"copy", `var src, dst []byte; copy(dst, src)`, `func([]byte, []byte) int`},
	{"copy", `type T [][]int; var src, dst T; _ = copy(dst, src)`, `func(p.T, p.T) int`},
	{"copy", `var src string; var dst []byte; copy(dst, src)`, `func([]byte, string) int`},
	{"copy", `type T string; type U []byte; var src T; var dst U; copy(dst, src)`, `func(p.U, p.T) int`},
	{"copy", `var dst []byte; copy(dst, "hello")`, `func([]byte, string) int`},

	{"delete", `var m map[string]bool; delete(m, "foo")`, `func(map[string]bool, string)`},
	{"delete", `type (K string; V int); var m map[K]V; delete(m, "foo")`, `func(map[p.K]p.V, p.K)`},

	{"imag", `_ = imag(1i)`, `invalid type`}, // constant
	{"imag", `var c complex64; _ = imag(c)`, `func(complex64) float32`},
	{"imag", `var c complex128; _ = imag(c)`, `func(complex128) float64`},
	{"imag", `type C64 complex64; var c C64; _ = imag(c)`, `func(p.C64) float32`},
	{"imag", `type C128 complex128; var c C128; _ = imag(c)`, `func(p.C128) float64`},

	{"real", `_ = real(1i)`, `invalid type`}, // constant
	{"real", `var c complex64; _ = real(c)`, `func(complex64) float32`},
	{"real", `var c complex128; _ = real(c)`, `func(complex128) float64`},
	{"real", `type C64 complex64; var c C64; _ = real(c)`, `func(p.C64) float32`},
	{"real", `type C128 complex128; var c C128; _ = real(c)`, `func(p.C128) float64`},

	{"make", `_ = make([]int, 10)`, `func([]int, int) []int`},
	{"make", `type T []byte; _ = make(T, 10, 20)`, `func(p.T, int, int) p.T`},

	// issue #37349
	{"make", `              _ = make([]int, 0   )`, `func([]int, int) []int`},
	{"make", `var l    int; _ = make([]int, l   )`, `func([]int, int) []int`},
	{"make", `              _ = make([]int, 0, 0)`, `func([]int, int, int) []int`},
	{"make", `var l    int; _ = make([]int, l, 0)`, `func([]int, int, int) []int`},
	{"make", `var    c int; _ = make([]int, 0, c)`, `func([]int, int, int) []int`},
	{"make", `var l, c int; _ = make([]int, l, c)`, `func([]int, int, int) []int`},

	// issue #37393
	{"make", `                _ = make([]int       , 0   )`, `func([]int, int) []int`},
	{"make", `var l    byte ; _ = make([]int8      , l   )`, `func([]int8, byte) []int8`},
	{"make", `                _ = make([]int16     , 0, 0)`, `func([]int16, int, int) []int16`},
	{"make", `var l    int16; _ = make([]string    , l, 0)`, `func([]string, int16, int) []string`},
	{"make", `var    c int32; _ = make([]float64   , 0, c)`, `func([]float64, int, int32) []float64`},
	{"make", `var l, c uint ; _ = make([]complex128, l, c)`, `func([]complex128, uint, uint) []complex128`},

	// issue #45667
	{"make", `const l uint = 1; _ = make([]int, l)`, `func([]int, uint) []int`},

	{"new", `_ = new(int)`, `func(int) *int`},
	{"new", `type T struct{}; _ = new(T)`, `func(p.T) *p.T`},

	{"panic", `panic(0)`, `func(interface{})`},
	{"panic", `panic("foo")`, `func(interface{})`},

	{"print", `print()`, `func()`},
	{"print", `print(0)`, `func(int)`},
	{"print", `print(1, 2.0, "foo", true)`, `func(int, float64, string, bool)`},

	{"println", `println()`, `func()`},
	{"println", `println(0)`, `func(int)`},
	{"println", `println(1, 2.0, "foo", true)`, `func(int, float64, string, bool)`},

	{"recover", `recover()`, `func() interface{}`},
	{"recover", `_ = recover()`, `func() interface{}`},

	{"Add", `var p unsafe.Pointer; _ = unsafe.Add(p, -1.0)`, `func(unsafe.Pointer, int) unsafe.Pointer`},
	{"Add", `var p unsafe.Pointer; var n uintptr; _ = unsafe.Add(p, n)`, `func(unsafe.Pointer, uintptr) unsafe.Pointer`},
	{"Add", `_ = unsafe.Add(nil, 0)`, `func(unsafe.Pointer, int) unsafe.Pointer`},

	{"Alignof", `_ = unsafe.Alignof(0)`, `invalid type`},                 // constant
	{"Alignof", `var x struct{}; _ = unsafe.Alignof(x)`, `invalid type`}, // constant
	{"Alignof", `var x P; _ = unsafe.Alignof(x)`, `func(P) uintptr`},

	{"Offsetof", `var x struct{f bool}; _ = unsafe.Offsetof(x.f)`, `invalid type`},           // constant
	{"Offsetof", `var x struct{_ int; f bool}; _ = unsafe.Offsetof((&x).f)`, `invalid type`}, // constant
	{"Offsetof", `var x struct{_ int; f P}; _ = unsafe.Offsetof((&x).f)`, `func(P) uintptr`},

	{"Sizeof", `_ = unsafe.Sizeof(0)`, `invalid type`},                 // constant
	{"Sizeof", `var x struct{}; _ = unsafe.Sizeof(x)`, `invalid type`}, // constant
	{"Sizeof", `var x P; _ = unsafe.Sizeof(x)`, `func(P) uintptr`},

	{"Slice", `var p *int; _ = unsafe.Slice(p, 1)`, `func(*int, int) []int`},
	{"Slice", `var p *byte; var n uintptr; _ = unsafe.Slice(p, n)`, `func(*byte, uintptr) []byte`},

	{"SliceData", "var a []int; _ = unsafe.SliceData(a)", `func([]int) *int`},
	{"SliceData", "type sliceType []int; var a sliceType; _ = unsafe.SliceData(a)", `func([]int) *int`},

	{"String", `var p *byte; _ = unsafe.String(p, 1)`, `func(*byte, int) string`},
	{"String", `type pbyte *byte; var p  pbyte; var n uintptr; _ = unsafe.String(p, n)`, `func(*byte, uintptr) string`},

	{"StringData", `var s string; _ = unsafe.StringData(s)`, `func(string) *byte`},
	{"StringData", `var s = "abc"; _ = unsafe.StringData(s)`, `func(string) *byte`},

	{"assert", `assert(true)`, `invalid type`},                                    // constant
	{"assert", `type B bool; const pred B = 1 < 2; assert(pred)`, `invalid type`}, // constant

	// no tests for trace since it produces output as a side-effect
}

func TestBuiltinSignatures(t *testing.T) {
	DefPredeclaredTestFuncs()

	seen := map[string]bool{"trace": true} // no test for trace built-in; add it manually
	for _, call := range builtinCalls {
		testBuiltinSignature(t, call.name, call.src, call.sig)
		seen[call.name] = true
	}

	// make sure we didn't miss one
	for _, name := range Universe.Names() {
		if _, ok := Universe.Lookup(name).(*Builtin); ok && !seen[name] {
			t.Errorf("missing test for %s", name)
		}
	}
	for _, name := range Unsafe.Scope().Names() {
		if _, ok := Unsafe.Scope().Lookup(name).(*Builtin); ok && !seen[name] {
			t.Errorf("missing test for unsafe.%s", name)
		}
	}
}

func parseGenericSrc(path, src string) (*syntax.File, error) {
	errh := func(error) {} // dummy error handler so that parsing continues in presence of errors
	return syntax.Parse(syntax.NewFileBase(path), strings.NewReader(src), errh, nil, 0)
}

func testBuiltinSignature(t *testing.T, name, src0, want string) {
	src := fmt.Sprintf(`package p; import "unsafe"; type _ unsafe.Pointer /* use unsafe */; func _[P ~[]byte]() { %s }`, src0)
	f, err := parseGenericSrc("", src)
	if err != nil {
		t.Errorf("%s: %s", src0, err)
		return
	}

	conf := Config{Importer: defaultImporter()}
	uses := make(map[*syntax.Name]Object)
	types := make(map[syntax.Expr]TypeAndValue)
	_, err = conf.Check(f.PkgName.Value, []*syntax.File{f}, &Info{Uses: uses, Types: types})
	if err != nil {
		t.Errorf("%s: %s", src0, err)
		return
	}

	// find called function
	n := 0
	var fun syntax.Expr
	for x := range types {
		if call, _ := x.(*syntax.CallExpr); call != nil {
			fun = call.Fun
			n++
		}
	}
	if n != 1 {
		t.Errorf("%s: got %d CallExprs; want 1", src0, n)
		return
	}

	// check recorded types for fun and descendents (may be parenthesized)
	for {
		// the recorded type for the built-in must match the wanted signature
		typ := types[fun].Type
		if typ == nil {
			t.Errorf("%s: no type recorded for %s", src0, syntax.String(fun))
			return
		}
		if got := typ.String(); got != want {
			t.Errorf("%s: got type %s; want %s", src0, got, want)
			return
		}

		// called function must be a (possibly parenthesized, qualified)
		// identifier denoting the expected built-in
		switch p := fun.(type) {
		case *syntax.Name:
			obj := uses[p]
			if obj == nil {
				t.Errorf("%s: no object found for %s", src0, p.Value)
				return
			}
			bin, _ := obj.(*Builtin)
			if bin == nil {
				t.Errorf("%s: %s does not denote a built-in", src0, p.Value)
				return
			}
			if bin.Name() != name {
				t.Errorf("%s: got built-in %s; want %s", src0, bin.Name(), name)
				return
			}
			return // we're done

		case *syntax.ParenExpr:
			fun = p.X // unpack

		case *syntax.SelectorExpr:
			// built-in from package unsafe - ignore details
			return // we're done

		default:
			t.Errorf("%s: invalid function call", src0)
			return
		}
	}
}
