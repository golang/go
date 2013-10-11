// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/parser"
	"testing"
)

var builtinCalls = []struct {
	id  builtinId
	src string
	sig string
}{
	{_Append, `var s []int; _ = append(s)`, `func([]int, ...int) []int`},
	{_Append, `var s []int; _ = append(s, 0)`, `func([]int, ...int) []int`},
	{_Append, `var s []int; _ = (append)(s, 0)`, `func([]int, ...int) []int`},
	{_Append, `var s []byte; _ = ((append))(s, 0)`, `func([]byte, ...byte) []byte`},
	// Note that ...uint8 (instead of ..byte) appears below because that is the type
	// that corresponds to Typ[byte] (an alias) - in the other cases, the type name
	// is chosen by the source. Either way, byte and uint8 denote identical types.
	{_Append, `var s []byte; _ = append(s, "foo"...)`, `func([]byte, ...uint8) []byte`},
	{_Append, `type T []byte; var s T; _ = append(s, "foo"...)`, `func(p.T, ...uint8) p.T`},

	{_Cap, `var s [10]int; _ = cap(s)`, `invalid type`},  // constant
	{_Cap, `var s [10]int; _ = cap(&s)`, `invalid type`}, // constant
	{_Cap, `var s []int64; _ = cap(s)`, `func([]int64) int`},
	{_Cap, `var c chan<-bool; _ = cap(c)`, `func(chan<- bool) int`},

	{_Len, `_ = len("foo")`, `invalid type`}, // constant
	{_Len, `var s string; _ = len(s)`, `func(string) int`},
	{_Len, `var s [10]int; _ = len(s)`, `invalid type`},  // constant
	{_Len, `var s [10]int; _ = len(&s)`, `invalid type`}, // constant
	{_Len, `var s []int64; _ = len(s)`, `func([]int64) int`},
	{_Len, `var c chan<-bool; _ = len(c)`, `func(chan<- bool) int`},
	{_Len, `var m map[string]float32; _ = len(m)`, `func(map[string]float32) int`},

	{_Close, `var c chan int; close(c)`, `func(chan int)`},
	{_Close, `var c chan<- chan string; close(c)`, `func(chan<- chan string)`},

	{_Complex, `_ = complex(1, 0)`, `invalid type`}, // constant
	{_Complex, `var re float32; _ = complex(re, 1.0)`, `func(float32, float32) complex64`},
	{_Complex, `var im float64; _ = complex(1, im)`, `func(float64, float64) complex128`},
	{_Complex, `type F32 float32; var re, im F32; _ = complex(re, im)`, `func(p.F32, p.F32) complex64`},
	{_Complex, `type F64 float64; var re, im F64; _ = complex(re, im)`, `func(p.F64, p.F64) complex128`},

	{_Copy, `var src, dst []byte; copy(dst, src)`, `func([]byte, []byte) int`},
	{_Copy, `type T [][]int; var src, dst T; _ = copy(dst, src)`, `func([][]int, [][]int) int`},

	{_Delete, `var m map[string]bool; delete(m, "foo")`, `func(map[string]bool, string)`},
	{_Delete, `type (K string; V int); var m map[K]V; delete(m, "foo")`, `func(map[p.K]p.V, p.K)`},

	{_Imag, `_ = imag(1i)`, `invalid type`}, // constant
	{_Imag, `var c complex64; _ = imag(c)`, `func(complex64) float32`},
	{_Imag, `var c complex128; _ = imag(c)`, `func(complex128) float64`},
	{_Imag, `type C64 complex64; var c C64; _ = imag(c)`, `func(p.C64) float32`},
	{_Imag, `type C128 complex128; var c C128; _ = imag(c)`, `func(p.C128) float64`},

	{_Real, `_ = real(1i)`, `invalid type`}, // constant
	{_Real, `var c complex64; _ = real(c)`, `func(complex64) float32`},
	{_Real, `var c complex128; _ = real(c)`, `func(complex128) float64`},
	{_Real, `type C64 complex64; var c C64; _ = real(c)`, `func(p.C64) float32`},
	{_Real, `type C128 complex128; var c C128; _ = real(c)`, `func(p.C128) float64`},

	{_Make, `_ = make([]int, 10)`, `func([]int, int) []int`},
	{_Make, `type T []byte; _ = make(T, 10, 20)`, `func(p.T, int, int) p.T`},

	{_New, `_ = new(int)`, `func(int) *int`},
	{_New, `type T struct{}; _ = new(T)`, `func(p.T) *p.T`},

	{_Panic, `panic(0)`, `func(interface{})`},
	{_Panic, `panic("foo")`, `func(interface{})`},

	{_Print, `print()`, `func()`},
	{_Print, `print(0)`, `func(int)`},
	{_Print, `print(1, 2.0, "foo", true)`, `func(int, float64, string, bool)`},

	{_Println, `println()`, `func()`},
	{_Println, `println(0)`, `func(int)`},
	{_Println, `println(1, 2.0, "foo", true)`, `func(int, float64, string, bool)`},

	{_Recover, `recover()`, `func() interface{}`},
	{_Recover, `_ = recover()`, `func() interface{}`},

	{_Alignof, `_ = unsafe.Alignof(0)`, `invalid type`},                 // constant
	{_Alignof, `var x struct{}; _ = unsafe.Alignof(x)`, `invalid type`}, // constant

	{_Offsetof, `var x struct{f bool}; _ = unsafe.Offsetof(x.f)`, `invalid type`},           // constant
	{_Offsetof, `var x struct{_ int; f bool}; _ = unsafe.Offsetof((&x).f)`, `invalid type`}, // constant

	{_Sizeof, `_ = unsafe.Sizeof(0)`, `invalid type`},                 // constant
	{_Sizeof, `var x struct{}; _ = unsafe.Sizeof(x)`, `invalid type`}, // constant

	{_Assert, `assert(true)`, `invalid type`},                                    // constant
	{_Assert, `type B bool; const pred B = 1 < 2; assert(pred)`, `invalid type`}, // constant

	// no tests for trace since it produces output as a side-effect
}

func TestBuiltinSignatures(t *testing.T) {
	defPredeclaredTestFuncs()

	seen := map[builtinId]bool{_Trace: true} // no test for _Trace; add it manually
	for _, call := range builtinCalls {
		testBuiltinSignature(t, call.id, call.src, call.sig)
		seen[call.id] = true
	}

	// make sure we didn't miss one
	for i := range predeclaredFuncs {
		if id := builtinId(i); !seen[id] {
			t.Errorf("missing test for %s", predeclaredFuncs[id].name)
		}
	}
}

func testBuiltinSignature(t *testing.T, id builtinId, src0, want string) {
	src := fmt.Sprintf(`package p; import "unsafe"; type _ unsafe.Pointer /* use unsafe */; func _() { %s }`, src0)
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Errorf("%s: %s", src0, err)
		return
	}

	var conf Config
	objects := make(map[*ast.Ident]Object)
	types := make(map[ast.Expr]Type)
	_, err = conf.Check(f.Name.Name, fset, []*ast.File{f}, &Info{Objects: objects, Types: types})
	if err != nil {
		t.Errorf("%s: %s", src0, err)
		return
	}

	// find called function
	n := 0
	var fun ast.Expr
	for x, _ := range types {
		if call, _ := x.(*ast.CallExpr); call != nil {
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
		typ := types[fun]
		if typ == nil {
			t.Errorf("%s: no type recorded for %s", src0, exprString(fun))
			return
		}
		if got := typ.String(); got != want {
			t.Errorf("%s: got type %s; want %s", src0, got, want)
			return
		}

		// called function must be a (possibly parenthesized, qualified)
		// identifier denoting the expected built-in
		switch p := fun.(type) {
		case *ast.Ident:
			obj := objects[p]
			if obj == nil {
				t.Errorf("%s: no object found for %s", src0, p)
				return
			}
			bin, _ := obj.(*Builtin)
			if bin == nil {
				t.Errorf("%s: %s does not denote a built-in", src0, p)
				return
			}
			if bin.id != id {
				t.Errorf("%s: got built-in %s; want %s", src0, bin.name, predeclaredFuncs[id].name)
				return
			}
			return // we're done

		case *ast.ParenExpr:
			fun = p.X // unpack

		case *ast.SelectorExpr:
			// built-in from package unsafe - ignore details
			return // we're done

		default:
			t.Errorf("%s: invalid function call", src0)
			return
		}
	}
}
