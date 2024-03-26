// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"errors"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"internal/goversion"
	"internal/testenv"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"

	. "go/types"
)

// nopos indicates an unknown position
var nopos token.Pos

func mustParse(fset *token.FileSet, src string) *ast.File {
	f, err := parser.ParseFile(fset, pkgName(src), src, parser.ParseComments)
	if err != nil {
		panic(err) // so we don't need to pass *testing.T
	}
	return f
}

func typecheck(src string, conf *Config, info *Info) (*Package, error) {
	fset := token.NewFileSet()
	f := mustParse(fset, src)
	if conf == nil {
		conf = &Config{
			Error:    func(err error) {}, // collect all errors
			Importer: importer.Default(),
		}
	}
	return conf.Check(f.Name.Name, fset, []*ast.File{f}, info)
}

func mustTypecheck(src string, conf *Config, info *Info) *Package {
	pkg, err := typecheck(src, conf, info)
	if err != nil {
		panic(err) // so we don't need to pass *testing.T
	}
	return pkg
}

// pkgName extracts the package name from src, which must contain a package header.
func pkgName(src string) string {
	const kw = "package "
	if i := strings.Index(src, kw); i >= 0 {
		after := src[i+len(kw):]
		n := len(after)
		if i := strings.IndexAny(after, "\n\t ;/"); i >= 0 {
			n = i
		}
		return after[:n]
	}
	panic("missing package header: " + src)
}

func TestValuesInfo(t *testing.T) {
	var tests = []struct {
		src  string
		expr string // constant expression
		typ  string // constant type
		val  string // constant value
	}{
		{`package a0; const _ = false`, `false`, `untyped bool`, `false`},
		{`package a1; const _ = 0`, `0`, `untyped int`, `0`},
		{`package a2; const _ = 'A'`, `'A'`, `untyped rune`, `65`},
		{`package a3; const _ = 0.`, `0.`, `untyped float`, `0`},
		{`package a4; const _ = 0i`, `0i`, `untyped complex`, `(0 + 0i)`},
		{`package a5; const _ = "foo"`, `"foo"`, `untyped string`, `"foo"`},

		{`package b0; var _ = false`, `false`, `bool`, `false`},
		{`package b1; var _ = 0`, `0`, `int`, `0`},
		{`package b2; var _ = 'A'`, `'A'`, `rune`, `65`},
		{`package b3; var _ = 0.`, `0.`, `float64`, `0`},
		{`package b4; var _ = 0i`, `0i`, `complex128`, `(0 + 0i)`},
		{`package b5; var _ = "foo"`, `"foo"`, `string`, `"foo"`},

		{`package c0a; var _ = bool(false)`, `false`, `bool`, `false`},
		{`package c0b; var _ = bool(false)`, `bool(false)`, `bool`, `false`},
		{`package c0c; type T bool; var _ = T(false)`, `T(false)`, `c0c.T`, `false`},

		{`package c1a; var _ = int(0)`, `0`, `int`, `0`},
		{`package c1b; var _ = int(0)`, `int(0)`, `int`, `0`},
		{`package c1c; type T int; var _ = T(0)`, `T(0)`, `c1c.T`, `0`},

		{`package c2a; var _ = rune('A')`, `'A'`, `rune`, `65`},
		{`package c2b; var _ = rune('A')`, `rune('A')`, `rune`, `65`},
		{`package c2c; type T rune; var _ = T('A')`, `T('A')`, `c2c.T`, `65`},

		{`package c3a; var _ = float32(0.)`, `0.`, `float32`, `0`},
		{`package c3b; var _ = float32(0.)`, `float32(0.)`, `float32`, `0`},
		{`package c3c; type T float32; var _ = T(0.)`, `T(0.)`, `c3c.T`, `0`},

		{`package c4a; var _ = complex64(0i)`, `0i`, `complex64`, `(0 + 0i)`},
		{`package c4b; var _ = complex64(0i)`, `complex64(0i)`, `complex64`, `(0 + 0i)`},
		{`package c4c; type T complex64; var _ = T(0i)`, `T(0i)`, `c4c.T`, `(0 + 0i)`},

		{`package c5a; var _ = string("foo")`, `"foo"`, `string`, `"foo"`},
		{`package c5b; var _ = string("foo")`, `string("foo")`, `string`, `"foo"`},
		{`package c5c; type T string; var _ = T("foo")`, `T("foo")`, `c5c.T`, `"foo"`},
		{`package c5d; var _ = string(65)`, `65`, `untyped int`, `65`},
		{`package c5e; var _ = string('A')`, `'A'`, `untyped rune`, `65`},
		{`package c5f; type T string; var _ = T('A')`, `'A'`, `untyped rune`, `65`},

		{`package d0; var _ = []byte("foo")`, `"foo"`, `string`, `"foo"`},
		{`package d1; var _ = []byte(string("foo"))`, `"foo"`, `string`, `"foo"`},
		{`package d2; var _ = []byte(string("foo"))`, `string("foo")`, `string`, `"foo"`},
		{`package d3; type T []byte; var _ = T("foo")`, `"foo"`, `string`, `"foo"`},

		{`package e0; const _ = float32( 1e-200)`, `float32(1e-200)`, `float32`, `0`},
		{`package e1; const _ = float32(-1e-200)`, `float32(-1e-200)`, `float32`, `0`},
		{`package e2; const _ = float64( 1e-2000)`, `float64(1e-2000)`, `float64`, `0`},
		{`package e3; const _ = float64(-1e-2000)`, `float64(-1e-2000)`, `float64`, `0`},
		{`package e4; const _ = complex64( 1e-200)`, `complex64(1e-200)`, `complex64`, `(0 + 0i)`},
		{`package e5; const _ = complex64(-1e-200)`, `complex64(-1e-200)`, `complex64`, `(0 + 0i)`},
		{`package e6; const _ = complex128( 1e-2000)`, `complex128(1e-2000)`, `complex128`, `(0 + 0i)`},
		{`package e7; const _ = complex128(-1e-2000)`, `complex128(-1e-2000)`, `complex128`, `(0 + 0i)`},

		{`package f0 ; var _ float32 =  1e-200`, `1e-200`, `float32`, `0`},
		{`package f1 ; var _ float32 = -1e-200`, `-1e-200`, `float32`, `0`},
		{`package f2a; var _ float64 =  1e-2000`, `1e-2000`, `float64`, `0`},
		{`package f3a; var _ float64 = -1e-2000`, `-1e-2000`, `float64`, `0`},
		{`package f2b; var _         =  1e-2000`, `1e-2000`, `float64`, `0`},
		{`package f3b; var _         = -1e-2000`, `-1e-2000`, `float64`, `0`},
		{`package f4 ; var _ complex64  =  1e-200 `, `1e-200`, `complex64`, `(0 + 0i)`},
		{`package f5 ; var _ complex64  = -1e-200 `, `-1e-200`, `complex64`, `(0 + 0i)`},
		{`package f6a; var _ complex128 =  1e-2000i`, `1e-2000i`, `complex128`, `(0 + 0i)`},
		{`package f7a; var _ complex128 = -1e-2000i`, `-1e-2000i`, `complex128`, `(0 + 0i)`},
		{`package f6b; var _            =  1e-2000i`, `1e-2000i`, `complex128`, `(0 + 0i)`},
		{`package f7b; var _            = -1e-2000i`, `-1e-2000i`, `complex128`, `(0 + 0i)`},

		{`package g0; const (a = len([iota]int{}); b; c); const _ = c`, `c`, `int`, `2`}, // go.dev/issue/22341
		{`package g1; var(j int32; s int; n = 1.0<<s == j)`, `1.0`, `int32`, `1`},        // go.dev/issue/48422
	}

	for _, test := range tests {
		info := Info{
			Types: make(map[ast.Expr]TypeAndValue),
		}
		name := mustTypecheck(test.src, nil, &info).Name()

		// look for expression
		var expr ast.Expr
		for e := range info.Types {
			if ExprString(e) == test.expr {
				expr = e
				break
			}
		}
		if expr == nil {
			t.Errorf("package %s: no expression found for %s", name, test.expr)
			continue
		}
		tv := info.Types[expr]

		// check that type is correct
		if got := tv.Type.String(); got != test.typ {
			t.Errorf("package %s: got type %s; want %s", name, got, test.typ)
			continue
		}

		// if we have a constant, check that value is correct
		if tv.Value != nil {
			if got := tv.Value.ExactString(); got != test.val {
				t.Errorf("package %s: got value %s; want %s", name, got, test.val)
			}
		} else {
			if test.val != "" {
				t.Errorf("package %s: no constant found; want %s", name, test.val)
			}
		}
	}
}

func TestTypesInfo(t *testing.T) {
	// Test sources that are not expected to typecheck must start with the broken prefix.
	const broken = "package broken_"

	var tests = []struct {
		src  string
		expr string // expression
		typ  string // value type
	}{
		// single-valued expressions of untyped constants
		{`package b0; var x interface{} = false`, `false`, `bool`},
		{`package b1; var x interface{} = 0`, `0`, `int`},
		{`package b2; var x interface{} = 0.`, `0.`, `float64`},
		{`package b3; var x interface{} = 0i`, `0i`, `complex128`},
		{`package b4; var x interface{} = "foo"`, `"foo"`, `string`},

		// uses of nil
		{`package n0; var _ *int = nil`, `nil`, `untyped nil`},
		{`package n1; var _ func() = nil`, `nil`, `untyped nil`},
		{`package n2; var _ []byte = nil`, `nil`, `untyped nil`},
		{`package n3; var _ map[int]int = nil`, `nil`, `untyped nil`},
		{`package n4; var _ chan int = nil`, `nil`, `untyped nil`},
		{`package n5; var _ interface{} = nil`, `nil`, `untyped nil`},
		{`package n6; import "unsafe"; var _ unsafe.Pointer = nil`, `nil`, `untyped nil`},

		{`package n10; var (x *int; _ = x == nil)`, `nil`, `untyped nil`},
		{`package n11; var (x func(); _ = x == nil)`, `nil`, `untyped nil`},
		{`package n12; var (x []byte; _ = x == nil)`, `nil`, `untyped nil`},
		{`package n13; var (x map[int]int; _ = x == nil)`, `nil`, `untyped nil`},
		{`package n14; var (x chan int; _ = x == nil)`, `nil`, `untyped nil`},
		{`package n15; var (x interface{}; _ = x == nil)`, `nil`, `untyped nil`},
		{`package n15; import "unsafe"; var (x unsafe.Pointer; _ = x == nil)`, `nil`, `untyped nil`},

		{`package n20; var _ = (*int)(nil)`, `nil`, `untyped nil`},
		{`package n21; var _ = (func())(nil)`, `nil`, `untyped nil`},
		{`package n22; var _ = ([]byte)(nil)`, `nil`, `untyped nil`},
		{`package n23; var _ = (map[int]int)(nil)`, `nil`, `untyped nil`},
		{`package n24; var _ = (chan int)(nil)`, `nil`, `untyped nil`},
		{`package n25; var _ = (interface{})(nil)`, `nil`, `untyped nil`},
		{`package n26; import "unsafe"; var _ = unsafe.Pointer(nil)`, `nil`, `untyped nil`},

		{`package n30; func f(*int) { f(nil) }`, `nil`, `untyped nil`},
		{`package n31; func f(func()) { f(nil) }`, `nil`, `untyped nil`},
		{`package n32; func f([]byte) { f(nil) }`, `nil`, `untyped nil`},
		{`package n33; func f(map[int]int) { f(nil) }`, `nil`, `untyped nil`},
		{`package n34; func f(chan int) { f(nil) }`, `nil`, `untyped nil`},
		{`package n35; func f(interface{}) { f(nil) }`, `nil`, `untyped nil`},
		{`package n35; import "unsafe"; func f(unsafe.Pointer) { f(nil) }`, `nil`, `untyped nil`},

		// comma-ok expressions
		{`package p0; var x interface{}; var _, _ = x.(int)`,
			`x.(int)`,
			`(int, bool)`,
		},
		{`package p1; var x interface{}; func _() { _, _ = x.(int) }`,
			`x.(int)`,
			`(int, bool)`,
		},
		{`package p2a; type mybool bool; var m map[string]complex128; var b mybool; func _() { _, b = m["foo"] }`,
			`m["foo"]`,
			`(complex128, p2a.mybool)`,
		},
		{`package p2b; var m map[string]complex128; var b bool; func _() { _, b = m["foo"] }`,
			`m["foo"]`,
			`(complex128, bool)`,
		},
		{`package p3; var c chan string; var _, _ = <-c`,
			`<-c`,
			`(string, bool)`,
		},

		// go.dev/issue/6796
		{`package issue6796_a; var x interface{}; var _, _ = (x.(int))`,
			`x.(int)`,
			`(int, bool)`,
		},
		{`package issue6796_b; var c chan string; var _, _ = (<-c)`,
			`(<-c)`,
			`(string, bool)`,
		},
		{`package issue6796_c; var c chan string; var _, _ = (<-c)`,
			`<-c`,
			`(string, bool)`,
		},
		{`package issue6796_d; var c chan string; var _, _ = ((<-c))`,
			`(<-c)`,
			`(string, bool)`,
		},
		{`package issue6796_e; func f(c chan string) { _, _ = ((<-c)) }`,
			`(<-c)`,
			`(string, bool)`,
		},

		// go.dev/issue/7060
		{`package issue7060_a; var ( m map[int]string; x, ok = m[0] )`,
			`m[0]`,
			`(string, bool)`,
		},
		{`package issue7060_b; var ( m map[int]string; x, ok interface{} = m[0] )`,
			`m[0]`,
			`(string, bool)`,
		},
		{`package issue7060_c; func f(x interface{}, ok bool, m map[int]string) { x, ok = m[0] }`,
			`m[0]`,
			`(string, bool)`,
		},
		{`package issue7060_d; var ( ch chan string; x, ok = <-ch )`,
			`<-ch`,
			`(string, bool)`,
		},
		{`package issue7060_e; var ( ch chan string; x, ok interface{} = <-ch )`,
			`<-ch`,
			`(string, bool)`,
		},
		{`package issue7060_f; func f(x interface{}, ok bool, ch chan string) { x, ok = <-ch }`,
			`<-ch`,
			`(string, bool)`,
		},

		// go.dev/issue/28277
		{`package issue28277_a; func f(...int)`,
			`...int`,
			`[]int`,
		},
		{`package issue28277_b; func f(a, b int, c ...[]struct{})`,
			`...[]struct{}`,
			`[][]struct{}`,
		},

		// go.dev/issue/47243
		{`package issue47243_a; var x int32; var _ = x << 3`, `3`, `untyped int`},
		{`package issue47243_b; var x int32; var _ = x << 3.`, `3.`, `untyped float`},
		{`package issue47243_c; var x int32; var _ = 1 << x`, `1 << x`, `int`},
		{`package issue47243_d; var x int32; var _ = 1 << x`, `1`, `int`},
		{`package issue47243_e; var x int32; var _ = 1 << 2`, `1`, `untyped int`},
		{`package issue47243_f; var x int32; var _ = 1 << 2`, `2`, `untyped int`},
		{`package issue47243_g; var x int32; var _ = int(1) << 2`, `2`, `untyped int`},
		{`package issue47243_h; var x int32; var _ = 1 << (2 << x)`, `1`, `int`},
		{`package issue47243_i; var x int32; var _ = 1 << (2 << x)`, `(2 << x)`, `untyped int`},
		{`package issue47243_j; var x int32; var _ = 1 << (2 << x)`, `2`, `untyped int`},

		// tests for broken code that doesn't type-check
		{broken + `x0; func _() { var x struct {f string}; x.f := 0 }`, `x.f`, `string`},
		{broken + `x1; func _() { var z string; type x struct {f string}; y := &x{q: z}}`, `z`, `string`},
		{broken + `x2; func _() { var a, b string; type x struct {f string}; z := &x{f: a, f: b,}}`, `b`, `string`},
		{broken + `x3; var x = panic("");`, `panic`, `func(interface{})`},
		{`package x4; func _() { panic("") }`, `panic`, `func(interface{})`},
		{broken + `x5; func _() { var x map[string][...]int; x = map[string][...]int{"": {1,2,3}} }`, `x`, `map[string]invalid type`},

		// parameterized functions
		{`package p0; func f[T any](T) {}; var _ = f[int]`, `f`, `func[T any](T)`},
		{`package p1; func f[T any](T) {}; var _ = f[int]`, `f[int]`, `func(int)`},
		{`package p2; func f[T any](T) {}; func _() { f(42) }`, `f`, `func(int)`},
		{`package p3; func f[T any](T) {}; func _() { f[int](42) }`, `f[int]`, `func(int)`},
		{`package p4; func f[T any](T) {}; func _() { f[int](42) }`, `f`, `func[T any](T)`},
		{`package p5; func f[T any](T) {}; func _() { f(42) }`, `f(42)`, `()`},

		// type parameters
		{`package t0; type t[] int; var _ t`, `t`, `t0.t`}, // t[] is a syntax error that is ignored in this test in favor of t
		{`package t1; type t[P any] int; var _ t[int]`, `t`, `t1.t[P any]`},
		{`package t2; type t[P interface{}] int; var _ t[int]`, `t`, `t2.t[P interface{}]`},
		{`package t3; type t[P, Q interface{}] int; var _ t[int, int]`, `t`, `t3.t[P, Q interface{}]`},
		{broken + `t4; type t[P, Q interface{ m() }] int; var _ t[int, int]`, `t`, `broken_t4.t[P, Q interface{m()}]`},

		// instantiated types must be sanitized
		{`package g0; type t[P any] int; var x struct{ f t[int] }; var _ = x.f`, `x.f`, `g0.t[int]`},

		// go.dev/issue/45096
		{`package issue45096; func _[T interface{ ~int8 | ~int16 | ~int32  }](x T) { _ = x < 0 }`, `0`, `T`},

		// go.dev/issue/47895
		{`package p; import "unsafe"; type S struct { f int }; var s S; var _ = unsafe.Offsetof(s.f)`, `s.f`, `int`},

		// go.dev/issue/50093
		{`package u0a; func _[_ interface{int}]() {}`, `int`, `int`},
		{`package u1a; func _[_ interface{~int}]() {}`, `~int`, `~int`},
		{`package u2a; func _[_ interface{int | string}]() {}`, `int | string`, `int | string`},
		{`package u3a; func _[_ interface{int | string | ~bool}]() {}`, `int | string | ~bool`, `int | string | ~bool`},
		{`package u3a; func _[_ interface{int | string | ~bool}]() {}`, `int | string`, `int | string`},
		{`package u3a; func _[_ interface{int | string | ~bool}]() {}`, `~bool`, `~bool`},
		{`package u3a; func _[_ interface{int | string | ~float64|~bool}]() {}`, `int | string | ~float64`, `int | string | ~float64`},

		{`package u0b; func _[_ int]() {}`, `int`, `int`},
		{`package u1b; func _[_ ~int]() {}`, `~int`, `~int`},
		{`package u2b; func _[_ int | string]() {}`, `int | string`, `int | string`},
		{`package u3b; func _[_ int | string | ~bool]() {}`, `int | string | ~bool`, `int | string | ~bool`},
		{`package u3b; func _[_ int | string | ~bool]() {}`, `int | string`, `int | string`},
		{`package u3b; func _[_ int | string | ~bool]() {}`, `~bool`, `~bool`},
		{`package u3b; func _[_ int | string | ~float64|~bool]() {}`, `int | string | ~float64`, `int | string | ~float64`},

		{`package u0c; type _ interface{int}`, `int`, `int`},
		{`package u1c; type _ interface{~int}`, `~int`, `~int`},
		{`package u2c; type _ interface{int | string}`, `int | string`, `int | string`},
		{`package u3c; type _ interface{int | string | ~bool}`, `int | string | ~bool`, `int | string | ~bool`},
		{`package u3c; type _ interface{int | string | ~bool}`, `int | string`, `int | string`},
		{`package u3c; type _ interface{int | string | ~bool}`, `~bool`, `~bool`},
		{`package u3c; type _ interface{int | string | ~float64|~bool}`, `int | string | ~float64`, `int | string | ~float64`},

		// reverse type inference
		{`package r1; var _ func(int) = g; func g[P any](P) {}`, `g`, `func(int)`},
		{`package r2; var _ func(int) = g[int]; func g[P any](P) {}`, `g`, `func[P any](P)`}, // go.dev/issues/60212
		{`package r3; var _ func(int) = g[int]; func g[P any](P) {}`, `g[int]`, `func(int)`},
		{`package r4; var _ func(int, string) = g; func g[P, Q any](P, Q) {}`, `g`, `func(int, string)`},
		{`package r5; var _ func(int, string) = g[int]; func g[P, Q any](P, Q) {}`, `g`, `func[P, Q any](P, Q)`}, // go.dev/issues/60212
		{`package r6; var _ func(int, string) = g[int]; func g[P, Q any](P, Q) {}`, `g[int]`, `func(int, string)`},

		{`package s1; func _() { f(g) }; func f(func(int)) {}; func g[P any](P) {}`, `g`, `func(int)`},
		{`package s2; func _() { f(g[int]) }; func f(func(int)) {}; func g[P any](P) {}`, `g`, `func[P any](P)`}, // go.dev/issues/60212
		{`package s3; func _() { f(g[int]) }; func f(func(int)) {}; func g[P any](P) {}`, `g[int]`, `func(int)`},
		{`package s4; func _() { f(g) }; func f(func(int, string)) {}; func g[P, Q any](P, Q) {}`, `g`, `func(int, string)`},
		{`package s5; func _() { f(g[int]) }; func f(func(int, string)) {}; func g[P, Q any](P, Q) {}`, `g`, `func[P, Q any](P, Q)`}, // go.dev/issues/60212
		{`package s6; func _() { f(g[int]) }; func f(func(int, string)) {}; func g[P, Q any](P, Q) {}`, `g[int]`, `func(int, string)`},

		{`package s7; func _() { f(g, h) }; func f[P any](func(int, P), func(P, string)) {}; func g[P any](P, P) {}; func h[P, Q any](P, Q) {}`, `g`, `func(int, int)`},
		{`package s8; func _() { f(g, h) }; func f[P any](func(int, P), func(P, string)) {}; func g[P any](P, P) {}; func h[P, Q any](P, Q) {}`, `h`, `func(int, string)`},
		{`package s9; func _() { f(g, h[int]) }; func f[P any](func(int, P), func(P, string)) {}; func g[P any](P, P) {}; func h[P, Q any](P, Q) {}`, `h`, `func[P, Q any](P, Q)`}, // go.dev/issues/60212
		{`package s10; func _() { f(g, h[int]) }; func f[P any](func(int, P), func(P, string)) {}; func g[P any](P, P) {}; func h[P, Q any](P, Q) {}`, `h[int]`, `func(int, string)`},
	}

	for _, test := range tests {
		info := Info{Types: make(map[ast.Expr]TypeAndValue)}
		var name string
		if strings.HasPrefix(test.src, broken) {
			pkg, err := typecheck(test.src, nil, &info)
			if err == nil {
				t.Errorf("package %s: expected to fail but passed", pkg.Name())
				continue
			}
			if pkg != nil {
				name = pkg.Name()
			}
		} else {
			name = mustTypecheck(test.src, nil, &info).Name()
		}

		// look for expression type
		var typ Type
		for e, tv := range info.Types {
			if ExprString(e) == test.expr {
				typ = tv.Type
				break
			}
		}
		if typ == nil {
			t.Errorf("package %s: no type found for %s", name, test.expr)
			continue
		}

		// check that type is correct
		if got := typ.String(); got != test.typ {
			t.Errorf("package %s: expr = %s: got %s; want %s", name, test.expr, got, test.typ)
		}
	}
}

func TestInstanceInfo(t *testing.T) {
	const lib = `package lib

func F[P any](P) {}

type T[P any] []P
`

	type testInst struct {
		name  string
		targs []string
		typ   string
	}

	var tests = []struct {
		src       string
		instances []testInst // recorded instances in source order
	}{
		{`package p0; func f[T any](T) {}; func _() { f(42) }`,
			[]testInst{{`f`, []string{`int`}, `func(int)`}},
		},
		{`package p1; func f[T any](T) T { panic(0) }; func _() { f('@') }`,
			[]testInst{{`f`, []string{`rune`}, `func(rune) rune`}},
		},
		{`package p2; func f[T any](...T) T { panic(0) }; func _() { f(0i) }`,
			[]testInst{{`f`, []string{`complex128`}, `func(...complex128) complex128`}},
		},
		{`package p3; func f[A, B, C any](A, *B, []C) {}; func _() { f(1.2, new(string), []byte{}) }`,
			[]testInst{{`f`, []string{`float64`, `string`, `byte`}, `func(float64, *string, []byte)`}},
		},
		{`package p4; func f[A, B any](A, *B, ...[]B) {}; func _() { f(1.2, new(byte)) }`,
			[]testInst{{`f`, []string{`float64`, `byte`}, `func(float64, *byte, ...[]byte)`}},
		},

		{`package s1; func f[T any, P interface{*T}](x T) {}; func _(x string) { f(x) }`,
			[]testInst{{`f`, []string{`string`, `*string`}, `func(x string)`}},
		},
		{`package s2; func f[T any, P interface{*T}](x []T) {}; func _(x []int) { f(x) }`,
			[]testInst{{`f`, []string{`int`, `*int`}, `func(x []int)`}},
		},
		{`package s3; type C[T any] interface{chan<- T}; func f[T any, P C[T]](x []T) {}; func _(x []int) { f(x) }`,
			[]testInst{
				{`C`, []string{`T`}, `interface{chan<- T}`},
				{`f`, []string{`int`, `chan<- int`}, `func(x []int)`},
			},
		},
		{`package s4; type C[T any] interface{chan<- T}; func f[T any, P C[T], Q C[[]*P]](x []T) {}; func _(x []int) { f(x) }`,
			[]testInst{
				{`C`, []string{`T`}, `interface{chan<- T}`},
				{`C`, []string{`[]*P`}, `interface{chan<- []*P}`},
				{`f`, []string{`int`, `chan<- int`, `chan<- []*chan<- int`}, `func(x []int)`},
			},
		},

		{`package t1; func f[T any, P interface{*T}]() T { panic(0) }; func _() { _ = f[string] }`,
			[]testInst{{`f`, []string{`string`, `*string`}, `func() string`}},
		},
		{`package t2; func f[T any, P interface{*T}]() T { panic(0) }; func _() { _ = (f[string]) }`,
			[]testInst{{`f`, []string{`string`, `*string`}, `func() string`}},
		},
		{`package t3; type C[T any] interface{chan<- T}; func f[T any, P C[T], Q C[[]*P]]() []T { return nil }; func _() { _ = f[int] }`,
			[]testInst{
				{`C`, []string{`T`}, `interface{chan<- T}`},
				{`C`, []string{`[]*P`}, `interface{chan<- []*P}`},
				{`f`, []string{`int`, `chan<- int`, `chan<- []*chan<- int`}, `func() []int`},
			},
		},
		{`package t4; type C[T any] interface{chan<- T}; func f[T any, P C[T], Q C[[]*P]]() []T { return nil }; func _() { _ = (f[int]) }`,
			[]testInst{
				{`C`, []string{`T`}, `interface{chan<- T}`},
				{`C`, []string{`[]*P`}, `interface{chan<- []*P}`},
				{`f`, []string{`int`, `chan<- int`, `chan<- []*chan<- int`}, `func() []int`},
			},
		},
		{`package i0; import "lib"; func _() { lib.F(42) }`,
			[]testInst{{`F`, []string{`int`}, `func(int)`}},
		},

		{`package duplfunc0; func f[T any](T) {}; func _() { f(42); f("foo"); f[int](3) }`,
			[]testInst{
				{`f`, []string{`int`}, `func(int)`},
				{`f`, []string{`string`}, `func(string)`},
				{`f`, []string{`int`}, `func(int)`},
			},
		},
		{`package duplfunc1; import "lib"; func _() { lib.F(42); lib.F("foo"); lib.F(3) }`,
			[]testInst{
				{`F`, []string{`int`}, `func(int)`},
				{`F`, []string{`string`}, `func(string)`},
				{`F`, []string{`int`}, `func(int)`},
			},
		},

		{`package type0; type T[P interface{~int}] struct{ x P }; var _ T[int]`,
			[]testInst{{`T`, []string{`int`}, `struct{x int}`}},
		},
		{`package type1; type T[P interface{~int}] struct{ x P }; var _ (T[int])`,
			[]testInst{{`T`, []string{`int`}, `struct{x int}`}},
		},
		{`package type2; type T[P interface{~int}] struct{ x P }; var _ T[(int)]`,
			[]testInst{{`T`, []string{`int`}, `struct{x int}`}},
		},
		{`package type3; type T[P1 interface{~[]P2}, P2 any] struct{ x P1; y P2 }; var _ T[[]int, int]`,
			[]testInst{{`T`, []string{`[]int`, `int`}, `struct{x []int; y int}`}},
		},
		{`package type4; import "lib"; var _ lib.T[int]`,
			[]testInst{{`T`, []string{`int`}, `[]int`}},
		},

		{`package dupltype0; type T[P interface{~int}] struct{ x P }; var x T[int]; var y T[int]`,
			[]testInst{
				{`T`, []string{`int`}, `struct{x int}`},
				{`T`, []string{`int`}, `struct{x int}`},
			},
		},
		{`package dupltype1; type T[P ~int] struct{ x P }; func (r *T[Q]) add(z T[Q]) { r.x += z.x }`,
			[]testInst{
				{`T`, []string{`Q`}, `struct{x Q}`},
				{`T`, []string{`Q`}, `struct{x Q}`},
			},
		},
		{`package dupltype1; import "lib"; var x lib.T[int]; var y lib.T[int]; var z lib.T[string]`,
			[]testInst{
				{`T`, []string{`int`}, `[]int`},
				{`T`, []string{`int`}, `[]int`},
				{`T`, []string{`string`}, `[]string`},
			},
		},
		{`package issue51803; func foo[T any](T) {}; func _() { foo[int]( /* leave arg away on purpose */ ) }`,
			[]testInst{{`foo`, []string{`int`}, `func(int)`}},
		},

		// reverse type inference
		{`package reverse1a; var f func(int) = g; func g[P any](P) {}`,
			[]testInst{{`g`, []string{`int`}, `func(int)`}},
		},
		{`package reverse1b; func f(func(int)) {}; func g[P any](P) {}; func _() { f(g) }`,
			[]testInst{{`g`, []string{`int`}, `func(int)`}},
		},
		{`package reverse2a; var f func(int, string) = g; func g[P, Q any](P, Q) {}`,
			[]testInst{{`g`, []string{`int`, `string`}, `func(int, string)`}},
		},
		{`package reverse2b; func f(func(int, string)) {}; func g[P, Q any](P, Q) {}; func _() { f(g) }`,
			[]testInst{{`g`, []string{`int`, `string`}, `func(int, string)`}},
		},
		{`package reverse2c; func f(func(int, string)) {}; func g[P, Q any](P, Q) {}; func _() { f(g[int]) }`,
			[]testInst{{`g`, []string{`int`, `string`}, `func(int, string)`}},
		},
		// reverse3a not possible (cannot assign to generic function outside of argument passing)
		{`package reverse3b; func f[R any](func(int) R) {}; func g[P any](P) string { return "" }; func _() { f(g) }`,
			[]testInst{
				{`f`, []string{`string`}, `func(func(int) string)`},
				{`g`, []string{`int`}, `func(int) string`},
			},
		},
		{`package reverse4a; var _, _ func([]int, *float32) = g, h; func g[P, Q any]([]P, *Q) {}; func h[R any]([]R, *float32) {}`,
			[]testInst{
				{`g`, []string{`int`, `float32`}, `func([]int, *float32)`},
				{`h`, []string{`int`}, `func([]int, *float32)`},
			},
		},
		{`package reverse4b; func f(_, _ func([]int, *float32)) {}; func g[P, Q any]([]P, *Q) {}; func h[R any]([]R, *float32) {}; func _() { f(g, h) }`,
			[]testInst{
				{`g`, []string{`int`, `float32`}, `func([]int, *float32)`},
				{`h`, []string{`int`}, `func([]int, *float32)`},
			},
		},
		{`package issue59956; func f(func(int), func(string), func(bool)) {}; func g[P any](P) {}; func _() { f(g, g, g) }`,
			[]testInst{
				{`g`, []string{`int`}, `func(int)`},
				{`g`, []string{`string`}, `func(string)`},
				{`g`, []string{`bool`}, `func(bool)`},
			},
		},
	}

	for _, test := range tests {
		imports := make(testImporter)
		conf := Config{Importer: imports}
		instMap := make(map[*ast.Ident]Instance)
		useMap := make(map[*ast.Ident]Object)
		makePkg := func(src string) *Package {
			pkg, err := typecheck(src, &conf, &Info{Instances: instMap, Uses: useMap})
			// allow error for issue51803
			if err != nil && (pkg == nil || pkg.Name() != "issue51803") {
				t.Fatal(err)
			}
			imports[pkg.Name()] = pkg
			return pkg
		}
		makePkg(lib)
		pkg := makePkg(test.src)

		t.Run(pkg.Name(), func(t *testing.T) {
			// Sort instances in source order for stability.
			instances := sortedInstances(instMap)
			if got, want := len(instances), len(test.instances); got != want {
				t.Fatalf("got %d instances, want %d", got, want)
			}

			// Pairwise compare with the expected instances.
			for ii, inst := range instances {
				var targs []Type
				for i := 0; i < inst.Inst.TypeArgs.Len(); i++ {
					targs = append(targs, inst.Inst.TypeArgs.At(i))
				}
				typ := inst.Inst.Type

				testInst := test.instances[ii]
				if got := inst.Ident.Name; got != testInst.name {
					t.Fatalf("got name %s, want %s", got, testInst.name)
				}
				if len(targs) != len(testInst.targs) {
					t.Fatalf("got %d type arguments; want %d", len(targs), len(testInst.targs))
				}
				for i, targ := range targs {
					if got := targ.String(); got != testInst.targs[i] {
						t.Errorf("type argument %d: got %s; want %s", i, got, testInst.targs[i])
					}
				}
				if got := typ.Underlying().String(); got != testInst.typ {
					t.Errorf("package %s: got %s; want %s", pkg.Name(), got, testInst.typ)
				}

				// Verify the invariant that re-instantiating the corresponding generic
				// type with TypeArgs results in an identical instance.
				ptype := useMap[inst.Ident].Type()
				lister, _ := ptype.(interface{ TypeParams() *TypeParamList })
				if lister == nil || lister.TypeParams().Len() == 0 {
					t.Fatalf("info.Types[%v] = %v, want parameterized type", inst.Ident, ptype)
				}
				inst2, err := Instantiate(nil, ptype, targs, true)
				if err != nil {
					t.Errorf("Instantiate(%v, %v) failed: %v", ptype, targs, err)
				}
				if !Identical(inst.Inst.Type, inst2) {
					t.Errorf("%v and %v are not identical", inst.Inst.Type, inst2)
				}
			}
		})
	}
}

type recordedInstance struct {
	Ident *ast.Ident
	Inst  Instance
}

func sortedInstances(m map[*ast.Ident]Instance) (instances []recordedInstance) {
	for id, inst := range m {
		instances = append(instances, recordedInstance{id, inst})
	}
	sort.Slice(instances, func(i, j int) bool {
		return CmpPos(instances[i].Ident.Pos(), instances[j].Ident.Pos()) < 0
	})
	return instances
}

func TestDefsInfo(t *testing.T) {
	var tests = []struct {
		src  string
		obj  string
		want string
	}{
		{`package p0; const x = 42`, `x`, `const p0.x untyped int`},
		{`package p1; const x int = 42`, `x`, `const p1.x int`},
		{`package p2; var x int`, `x`, `var p2.x int`},
		{`package p3; type x int`, `x`, `type p3.x int`},
		{`package p4; func f()`, `f`, `func p4.f()`},
		{`package p5; func f() int { x, _ := 1, 2; return x }`, `_`, `var _ int`},

		// Tests using generics.
		{`package g0; type x[T any] int`, `x`, `type g0.x[T any] int`},
		{`package g1; func f[T any]() {}`, `f`, `func g1.f[T any]()`},
		{`package g2; type x[T any] int; func (*x[_]) m() {}`, `m`, `func (*g2.x[_]).m()`},
	}

	for _, test := range tests {
		info := Info{
			Defs: make(map[*ast.Ident]Object),
		}
		name := mustTypecheck(test.src, nil, &info).Name()

		// find object
		var def Object
		for id, obj := range info.Defs {
			if id.Name == test.obj {
				def = obj
				break
			}
		}
		if def == nil {
			t.Errorf("package %s: %s not found", name, test.obj)
			continue
		}

		if got := def.String(); got != test.want {
			t.Errorf("package %s: got %s; want %s", name, got, test.want)
		}
	}
}

func TestUsesInfo(t *testing.T) {
	var tests = []struct {
		src  string
		obj  string
		want string
	}{
		{`package p0; func _() { _ = x }; const x = 42`, `x`, `const p0.x untyped int`},
		{`package p1; func _() { _ = x }; const x int = 42`, `x`, `const p1.x int`},
		{`package p2; func _() { _ = x }; var x int`, `x`, `var p2.x int`},
		{`package p3; func _() { type _ x }; type x int`, `x`, `type p3.x int`},
		{`package p4; func _() { _ = f }; func f()`, `f`, `func p4.f()`},

		// Tests using generics.
		{`package g0; func _[T any]() { _ = x }; const x = 42`, `x`, `const g0.x untyped int`},
		{`package g1; func _[T any](x T) { }`, `T`, `type parameter T any`},
		{`package g2; type N[A any] int; var _ N[int]`, `N`, `type g2.N[A any] int`},
		{`package g3; type N[A any] int; func (N[_]) m() {}`, `N`, `type g3.N[A any] int`},

		// Uses of fields are instantiated.
		{`package s1; type N[A any] struct{ a A }; var f = N[int]{}.a`, `a`, `field a int`},
		{`package s1; type N[A any] struct{ a A }; func (r N[B]) m(b B) { r.a = b }`, `a`, `field a B`},

		// Uses of methods are uses of the instantiated method.
		{`package m0; type N[A any] int; func (r N[B]) m() { r.n() }; func (N[C]) n() {}`, `n`, `func (m0.N[B]).n()`},
		{`package m1; type N[A any] int; func (r N[B]) m() { }; var f = N[int].m`, `m`, `func (m1.N[int]).m()`},
		{`package m2; func _[A any](v interface{ m() A }) { v.m() }`, `m`, `func (interface).m() A`},
		{`package m3; func f[A any]() interface{ m() A } { return nil }; var _ = f[int]().m()`, `m`, `func (interface).m() int`},
		{`package m4; type T[A any] func() interface{ m() A }; var x T[int]; var y = x().m`, `m`, `func (interface).m() int`},
		{`package m5; type T[A any] interface{ m() A }; func _[B any](t T[B]) { t.m() }`, `m`, `func (m5.T[B]).m() B`},
		{`package m6; type T[A any] interface{ m() }; func _[B any](t T[B]) { t.m() }`, `m`, `func (m6.T[B]).m()`},
		{`package m7; type T[A any] interface{ m() A }; func _(t T[int]) { t.m() }`, `m`, `func (m7.T[int]).m() int`},
		{`package m8; type T[A any] interface{ m() }; func _(t T[int]) { t.m() }`, `m`, `func (m8.T[int]).m()`},
		{`package m9; type T[A any] interface{ m() }; func _(t T[int]) { _ = t.m }`, `m`, `func (m9.T[int]).m()`},
		{
			`package m10; type E[A any] interface{ m() }; type T[B any] interface{ E[B]; n() }; func _(t T[int]) { t.m() }`,
			`m`,
			`func (m10.E[int]).m()`,
		},
		{`package m11; type T[A any] interface{ m(); n() }; func _(t1 T[int], t2 T[string]) { t1.m(); t2.n() }`, `m`, `func (m11.T[int]).m()`},
		{`package m12; type T[A any] interface{ m(); n() }; func _(t1 T[int], t2 T[string]) { t1.m(); t2.n() }`, `n`, `func (m12.T[string]).n()`},
	}

	for _, test := range tests {
		info := Info{
			Uses: make(map[*ast.Ident]Object),
		}
		name := mustTypecheck(test.src, nil, &info).Name()

		// find object
		var use Object
		for id, obj := range info.Uses {
			if id.Name == test.obj {
				if use != nil {
					panic(fmt.Sprintf("multiple uses of %q", id.Name))
				}
				use = obj
			}
		}
		if use == nil {
			t.Errorf("package %s: %s not found", name, test.obj)
			continue
		}

		if got := use.String(); got != test.want {
			t.Errorf("package %s: got %s; want %s", name, got, test.want)
		}
	}
}

func TestGenericMethodInfo(t *testing.T) {
	src := `package p

type N[A any] int

func (r N[B]) m() { r.m(); r.n() }

func (r *N[C]) n() {  }
`
	fset := token.NewFileSet()
	f := mustParse(fset, src)
	info := Info{
		Defs:       make(map[*ast.Ident]Object),
		Uses:       make(map[*ast.Ident]Object),
		Selections: make(map[*ast.SelectorExpr]*Selection),
	}
	var conf Config
	pkg, err := conf.Check("p", fset, []*ast.File{f}, &info)
	if err != nil {
		t.Fatal(err)
	}

	N := pkg.Scope().Lookup("N").Type().(*Named)

	// Find the generic methods stored on N.
	gm, gn := N.Method(0), N.Method(1)
	if gm.Name() == "n" {
		gm, gn = gn, gm
	}

	// Collect objects from info.
	var dm, dn *Func   // the declared methods
	var dmm, dmn *Func // the methods used in the body of m
	for _, decl := range f.Decls {
		fdecl, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		def := info.Defs[fdecl.Name].(*Func)
		switch fdecl.Name.Name {
		case "m":
			dm = def
			ast.Inspect(fdecl.Body, func(n ast.Node) bool {
				if call, ok := n.(*ast.CallExpr); ok {
					sel := call.Fun.(*ast.SelectorExpr)
					use := info.Uses[sel.Sel].(*Func)
					selection := info.Selections[sel]
					if selection.Kind() != MethodVal {
						t.Errorf("Selection kind = %v, want %v", selection.Kind(), MethodVal)
					}
					if selection.Obj() != use {
						t.Errorf("info.Selections contains %v, want %v", selection.Obj(), use)
					}
					switch sel.Sel.Name {
					case "m":
						dmm = use
					case "n":
						dmn = use
					}
				}
				return true
			})
		case "n":
			dn = def
		}
	}

	if gm != dm {
		t.Errorf(`N.Method(...) returns %v for "m", but Info.Defs has %v`, gm, dm)
	}
	if gn != dn {
		t.Errorf(`N.Method(...) returns %v for "m", but Info.Defs has %v`, gm, dm)
	}
	if dmm != dm {
		t.Errorf(`Inside "m", r.m uses %v, want the defined func %v`, dmm, dm)
	}
	if dmn == dn {
		t.Errorf(`Inside "m", r.n uses %v, want a func distinct from %v`, dmm, dm)
	}
}

func TestImplicitsInfo(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	var tests = []struct {
		src  string
		want string
	}{
		{`package p2; import . "fmt"; var _ = Println`, ""},           // no Implicits entry
		{`package p0; import local "fmt"; var _ = local.Println`, ""}, // no Implicits entry
		{`package p1; import "fmt"; var _ = fmt.Println`, "importSpec: package fmt"},

		{`package p3; func f(x interface{}) { switch x.(type) { case int: } }`, ""}, // no Implicits entry
		{`package p4; func f(x interface{}) { switch t := x.(type) { case int: _ = t } }`, "caseClause: var t int"},
		{`package p5; func f(x interface{}) { switch t := x.(type) { case int, uint: _ = t } }`, "caseClause: var t interface{}"},
		{`package p6; func f(x interface{}) { switch t := x.(type) { default: _ = t } }`, "caseClause: var t interface{}"},

		{`package p7; func f(x int) {}`, ""}, // no Implicits entry
		{`package p8; func f(int) {}`, "field: var  int"},
		{`package p9; func f() (complex64) { return 0 }`, "field: var  complex64"},
		{`package p10; type T struct{}; func (*T) f() {}`, "field: var  *p10.T"},

		// Tests using generics.
		{`package f0; func f[T any](x int) {}`, ""}, // no Implicits entry
		{`package f1; func f[T any](int) {}`, "field: var  int"},
		{`package f2; func f[T any](T) {}`, "field: var  T"},
		{`package f3; func f[T any]() (complex64) { return 0 }`, "field: var  complex64"},
		{`package f4; func f[T any](t T) (T) { return t }`, "field: var  T"},
		{`package t0; type T[A any] struct{}; func (*T[_]) f() {}`, "field: var  *t0.T[_]"},
		{`package t1; type T[A any] struct{}; func _(x interface{}) { switch t := x.(type) { case T[int]: _ = t } }`, "caseClause: var t t1.T[int]"},
		{`package t2; type T[A any] struct{}; func _[P any](x interface{}) { switch t := x.(type) { case T[P]: _ = t } }`, "caseClause: var t t2.T[P]"},
		{`package t3; func _[P any](x interface{}) { switch t := x.(type) { case P: _ = t } }`, "caseClause: var t P"},
	}

	for _, test := range tests {
		info := Info{
			Implicits: make(map[ast.Node]Object),
		}
		name := mustTypecheck(test.src, nil, &info).Name()

		// the test cases expect at most one Implicits entry
		if len(info.Implicits) > 1 {
			t.Errorf("package %s: %d Implicits entries found", name, len(info.Implicits))
			continue
		}

		// extract Implicits entry, if any
		var got string
		for n, obj := range info.Implicits {
			switch x := n.(type) {
			case *ast.ImportSpec:
				got = "importSpec"
			case *ast.CaseClause:
				got = "caseClause"
			case *ast.Field:
				got = "field"
			default:
				t.Fatalf("package %s: unexpected %T", name, x)
			}
			got += ": " + obj.String()
		}

		// verify entry
		if got != test.want {
			t.Errorf("package %s: got %q; want %q", name, got, test.want)
		}
	}
}

func TestPkgNameOf(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const src = `
package p

import (
	. "os"
	_ "io"
	"math"
	"path/filepath"
	snort "sort"
)

// avoid imported and not used errors
var (
	_ = Open // os.Open
	_ = math.Sin
	_ = filepath.Abs
	_ = snort.Ints
)
`

	var tests = []struct {
		path string // path string enclosed in "'s
		want string
	}{
		{`"os"`, "."},
		{`"io"`, "_"},
		{`"math"`, "math"},
		{`"path/filepath"`, "filepath"},
		{`"sort"`, "snort"},
	}

	fset := token.NewFileSet()
	f := mustParse(fset, src)
	info := Info{
		Defs:      make(map[*ast.Ident]Object),
		Implicits: make(map[ast.Node]Object),
	}
	var conf Config
	conf.Importer = importer.Default()
	_, err := conf.Check("p", fset, []*ast.File{f}, &info)
	if err != nil {
		t.Fatal(err)
	}

	// map import paths to importDecl
	imports := make(map[string]*ast.ImportSpec)
	for _, s := range f.Decls[0].(*ast.GenDecl).Specs {
		if imp, _ := s.(*ast.ImportSpec); imp != nil {
			imports[imp.Path.Value] = imp
		}
	}

	for _, test := range tests {
		imp := imports[test.path]
		if imp == nil {
			t.Fatalf("invalid test case: import path %s not found", test.path)
		}
		got := info.PkgNameOf(imp)
		if got == nil {
			t.Fatalf("import %s: package name not found", test.path)
		}
		if got.Name() != test.want {
			t.Errorf("import %s: got %s; want %s", test.path, got.Name(), test.want)
		}
	}

	// test non-existing importDecl
	if got := info.PkgNameOf(new(ast.ImportSpec)); got != nil {
		t.Errorf("got %s for non-existing import declaration", got.Name())
	}
}

func predString(tv TypeAndValue) string {
	var buf strings.Builder
	pred := func(b bool, s string) {
		if b {
			if buf.Len() > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(s)
		}
	}

	pred(tv.IsVoid(), "void")
	pred(tv.IsType(), "type")
	pred(tv.IsBuiltin(), "builtin")
	pred(tv.IsValue() && tv.Value != nil, "const")
	pred(tv.IsValue() && tv.Value == nil, "value")
	pred(tv.IsNil(), "nil")
	pred(tv.Addressable(), "addressable")
	pred(tv.Assignable(), "assignable")
	pred(tv.HasOk(), "hasOk")

	if buf.Len() == 0 {
		return "invalid"
	}
	return buf.String()
}

func TestPredicatesInfo(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	var tests = []struct {
		src  string
		expr string
		pred string
	}{
		// void
		{`package n0; func f() { f() }`, `f()`, `void`},

		// types
		{`package t0; type _ int`, `int`, `type`},
		{`package t1; type _ []int`, `[]int`, `type`},
		{`package t2; type _ func()`, `func()`, `type`},
		{`package t3; type _ func(int)`, `int`, `type`},
		{`package t3; type _ func(...int)`, `...int`, `type`},

		// built-ins
		{`package b0; var _ = len("")`, `len`, `builtin`},
		{`package b1; var _ = (len)("")`, `(len)`, `builtin`},

		// constants
		{`package c0; var _ = 42`, `42`, `const`},
		{`package c1; var _ = "foo" + "bar"`, `"foo" + "bar"`, `const`},
		{`package c2; const (i = 1i; _ = i)`, `i`, `const`},

		// values
		{`package v0; var (a, b int; _ = a + b)`, `a + b`, `value`},
		{`package v1; var _ = &[]int{1}`, `[]int{…}`, `value`},
		{`package v2; var _ = func(){}`, `(func() literal)`, `value`},
		{`package v4; func f() { _ = f }`, `f`, `value`},
		{`package v3; var _ *int = nil`, `nil`, `value, nil`},
		{`package v3; var _ *int = (nil)`, `(nil)`, `value, nil`},

		// addressable (and thus assignable) operands
		{`package a0; var (x int; _ = x)`, `x`, `value, addressable, assignable`},
		{`package a1; var (p *int; _ = *p)`, `*p`, `value, addressable, assignable`},
		{`package a2; var (s []int; _ = s[0])`, `s[0]`, `value, addressable, assignable`},
		{`package a3; var (s struct{f int}; _ = s.f)`, `s.f`, `value, addressable, assignable`},
		{`package a4; var (a [10]int; _ = a[0])`, `a[0]`, `value, addressable, assignable`},
		{`package a5; func _(x int) { _ = x }`, `x`, `value, addressable, assignable`},
		{`package a6; func _()(x int) { _ = x; return }`, `x`, `value, addressable, assignable`},
		{`package a7; type T int; func (x T) _() { _ = x }`, `x`, `value, addressable, assignable`},
		// composite literals are not addressable

		// assignable but not addressable values
		{`package s0; var (m map[int]int; _ = m[0])`, `m[0]`, `value, assignable, hasOk`},
		{`package s1; var (m map[int]int; _, _ = m[0])`, `m[0]`, `value, assignable, hasOk`},

		// hasOk expressions
		{`package k0; var (ch chan int; _ = <-ch)`, `<-ch`, `value, hasOk`},
		{`package k1; var (ch chan int; _, _ = <-ch)`, `<-ch`, `value, hasOk`},

		// missing entries
		// - package names are collected in the Uses map
		// - identifiers being declared are collected in the Defs map
		{`package m0; import "os"; func _() { _ = os.Stdout }`, `os`, `<missing>`},
		{`package m1; import p "os"; func _() { _ = p.Stdout }`, `p`, `<missing>`},
		{`package m2; const c = 0`, `c`, `<missing>`},
		{`package m3; type T int`, `T`, `<missing>`},
		{`package m4; var v int`, `v`, `<missing>`},
		{`package m5; func f() {}`, `f`, `<missing>`},
		{`package m6; func _(x int) {}`, `x`, `<missing>`},
		{`package m6; func _()(x int) { return }`, `x`, `<missing>`},
		{`package m6; type T int; func (x T) _() {}`, `x`, `<missing>`},
	}

	for _, test := range tests {
		info := Info{Types: make(map[ast.Expr]TypeAndValue)}
		name := mustTypecheck(test.src, nil, &info).Name()

		// look for expression predicates
		got := "<missing>"
		for e, tv := range info.Types {
			//println(name, ExprString(e))
			if ExprString(e) == test.expr {
				got = predString(tv)
				break
			}
		}

		if got != test.pred {
			t.Errorf("package %s: got %s; want %s", name, got, test.pred)
		}
	}
}

func TestScopesInfo(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	var tests = []struct {
		src    string
		scopes []string // list of scope descriptors of the form kind:varlist
	}{
		{`package p0`, []string{
			"file:",
		}},
		{`package p1; import ( "fmt"; m "math"; _ "os" ); var ( _ = fmt.Println; _ = m.Pi )`, []string{
			"file:fmt m",
		}},
		{`package p2; func _() {}`, []string{
			"file:", "func:",
		}},
		{`package p3; func _(x, y int) {}`, []string{
			"file:", "func:x y",
		}},
		{`package p4; func _(x, y int) { x, z := 1, 2; _ = z }`, []string{
			"file:", "func:x y z", // redeclaration of x
		}},
		{`package p5; func _(x, y int) (u, _ int) { return }`, []string{
			"file:", "func:u x y",
		}},
		{`package p6; func _() { { var x int; _ = x } }`, []string{
			"file:", "func:", "block:x",
		}},
		{`package p7; func _() { if true {} }`, []string{
			"file:", "func:", "if:", "block:",
		}},
		{`package p8; func _() { if x := 0; x < 0 { y := x; _ = y } }`, []string{
			"file:", "func:", "if:x", "block:y",
		}},
		{`package p9; func _() { switch x := 0; x {} }`, []string{
			"file:", "func:", "switch:x",
		}},
		{`package p10; func _() { switch x := 0; x { case 1: y := x; _ = y; default: }}`, []string{
			"file:", "func:", "switch:x", "case:y", "case:",
		}},
		{`package p11; func _(t interface{}) { switch t.(type) {} }`, []string{
			"file:", "func:t", "type switch:",
		}},
		{`package p12; func _(t interface{}) { switch t := t; t.(type) {} }`, []string{
			"file:", "func:t", "type switch:t",
		}},
		{`package p13; func _(t interface{}) { switch x := t.(type) { case int: _ = x } }`, []string{
			"file:", "func:t", "type switch:", "case:x", // x implicitly declared
		}},
		{`package p14; func _() { select{} }`, []string{
			"file:", "func:",
		}},
		{`package p15; func _(c chan int) { select{ case <-c: } }`, []string{
			"file:", "func:c", "comm:",
		}},
		{`package p16; func _(c chan int) { select{ case i := <-c: x := i; _ = x} }`, []string{
			"file:", "func:c", "comm:i x",
		}},
		{`package p17; func _() { for{} }`, []string{
			"file:", "func:", "for:", "block:",
		}},
		{`package p18; func _(n int) { for i := 0; i < n; i++ { _ = i } }`, []string{
			"file:", "func:n", "for:i", "block:",
		}},
		{`package p19; func _(a []int) { for i := range a { _ = i} }`, []string{
			"file:", "func:a", "range:i", "block:",
		}},
		{`package p20; var s int; func _(a []int) { for i, x := range a { s += x; _ = i } }`, []string{
			"file:", "func:a", "range:i x", "block:",
		}},
	}

	for _, test := range tests {
		info := Info{Scopes: make(map[ast.Node]*Scope)}
		name := mustTypecheck(test.src, nil, &info).Name()

		// number of scopes must match
		if len(info.Scopes) != len(test.scopes) {
			t.Errorf("package %s: got %d scopes; want %d", name, len(info.Scopes), len(test.scopes))
		}

		// scope descriptions must match
		for node, scope := range info.Scopes {
			kind := "<unknown node kind>"
			switch node.(type) {
			case *ast.File:
				kind = "file"
			case *ast.FuncType:
				kind = "func"
			case *ast.BlockStmt:
				kind = "block"
			case *ast.IfStmt:
				kind = "if"
			case *ast.SwitchStmt:
				kind = "switch"
			case *ast.TypeSwitchStmt:
				kind = "type switch"
			case *ast.CaseClause:
				kind = "case"
			case *ast.CommClause:
				kind = "comm"
			case *ast.ForStmt:
				kind = "for"
			case *ast.RangeStmt:
				kind = "range"
			}

			// look for matching scope description
			desc := kind + ":" + strings.Join(scope.Names(), " ")
			found := false
			for _, d := range test.scopes {
				if desc == d {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("package %s: no matching scope found for %s", name, desc)
			}
		}
	}
}

func TestInitOrderInfo(t *testing.T) {
	var tests = []struct {
		src   string
		inits []string
	}{
		{`package p0; var (x = 1; y = x)`, []string{
			"x = 1", "y = x",
		}},
		{`package p1; var (a = 1; b = 2; c = 3)`, []string{
			"a = 1", "b = 2", "c = 3",
		}},
		{`package p2; var (a, b, c = 1, 2, 3)`, []string{
			"a = 1", "b = 2", "c = 3",
		}},
		{`package p3; var _ = f(); func f() int { return 1 }`, []string{
			"_ = f()", // blank var
		}},
		{`package p4; var (a = 0; x = y; y = z; z = 0)`, []string{
			"a = 0", "z = 0", "y = z", "x = y",
		}},
		{`package p5; var (a, _ = m[0]; m map[int]string)`, []string{
			"a, _ = m[0]", // blank var
		}},
		{`package p6; var a, b = f(); func f() (_, _ int) { return z, z }; var z = 0`, []string{
			"z = 0", "a, b = f()",
		}},
		{`package p7; var (a = func() int { return b }(); b = 1)`, []string{
			"b = 1", "a = (func() int literal)()",
		}},
		{`package p8; var (a, b = func() (_, _ int) { return c, c }(); c = 1)`, []string{
			"c = 1", "a, b = (func() (_, _ int) literal)()",
		}},
		{`package p9; type T struct{}; func (T) m() int { _ = y; return 0 }; var x, y = T.m, 1`, []string{
			"y = 1", "x = T.m",
		}},
		{`package p10; var (d = c + b; a = 0; b = 0; c = 0)`, []string{
			"a = 0", "b = 0", "c = 0", "d = c + b",
		}},
		{`package p11; var (a = e + c; b = d + c; c = 0; d = 0; e = 0)`, []string{
			"c = 0", "d = 0", "b = d + c", "e = 0", "a = e + c",
		}},
		// emit an initializer for n:1 initializations only once (not for each node
		// on the lhs which may appear in different order in the dependency graph)
		{`package p12; var (a = x; b = 0; x, y = m[0]; m map[int]int)`, []string{
			"b = 0", "x, y = m[0]", "a = x",
		}},
		// test case from spec section on package initialization
		{`package p12

		var (
			a = c + b
			b = f()
			c = f()
			d = 3
		)

		func f() int {
			d++
			return d
		}`, []string{
			"d = 3", "b = f()", "c = f()", "a = c + b",
		}},
		// test case for go.dev/issue/7131
		{`package main

		var counter int
		func next() int { counter++; return counter }

		var _ = makeOrder()
		func makeOrder() []int { return []int{f, b, d, e, c, a} }

		var a       = next()
		var b, c    = next(), next()
		var d, e, f = next(), next(), next()
		`, []string{
			"a = next()", "b = next()", "c = next()", "d = next()", "e = next()", "f = next()", "_ = makeOrder()",
		}},
		// test case for go.dev/issue/10709
		{`package p13

		var (
		    v = t.m()
		    t = makeT(0)
		)

		type T struct{}

		func (T) m() int { return 0 }

		func makeT(n int) T {
		    if n > 0 {
		        return makeT(n-1)
		    }
		    return T{}
		}`, []string{
			"t = makeT(0)", "v = t.m()",
		}},
		// test case for go.dev/issue/10709: same as test before, but variable decls swapped
		{`package p14

		var (
		    t = makeT(0)
		    v = t.m()
		)

		type T struct{}

		func (T) m() int { return 0 }

		func makeT(n int) T {
		    if n > 0 {
		        return makeT(n-1)
		    }
		    return T{}
		}`, []string{
			"t = makeT(0)", "v = t.m()",
		}},
		// another candidate possibly causing problems with go.dev/issue/10709
		{`package p15

		var y1 = f1()

		func f1() int { return g1() }
		func g1() int { f1(); return x1 }

		var x1 = 0

		var y2 = f2()

		func f2() int { return g2() }
		func g2() int { return x2 }

		var x2 = 0`, []string{
			"x1 = 0", "y1 = f1()", "x2 = 0", "y2 = f2()",
		}},
	}

	for _, test := range tests {
		info := Info{}
		name := mustTypecheck(test.src, nil, &info).Name()

		// number of initializers must match
		if len(info.InitOrder) != len(test.inits) {
			t.Errorf("package %s: got %d initializers; want %d", name, len(info.InitOrder), len(test.inits))
			continue
		}

		// initializers must match
		for i, want := range test.inits {
			got := info.InitOrder[i].String()
			if got != want {
				t.Errorf("package %s, init %d: got %s; want %s", name, i, got, want)
				continue
			}
		}
	}
}

func TestMultiFileInitOrder(t *testing.T) {
	fset := token.NewFileSet()
	fileA := mustParse(fset, `package main; var a = 1`)
	fileB := mustParse(fset, `package main; var b = 2`)

	// The initialization order must not depend on the parse
	// order of the files, only on the presentation order to
	// the type-checker.
	for _, test := range []struct {
		files []*ast.File
		want  string
	}{
		{[]*ast.File{fileA, fileB}, "[a = 1 b = 2]"},
		{[]*ast.File{fileB, fileA}, "[b = 2 a = 1]"},
	} {
		var info Info
		if _, err := new(Config).Check("main", fset, test.files, &info); err != nil {
			t.Fatal(err)
		}
		if got := fmt.Sprint(info.InitOrder); got != test.want {
			t.Fatalf("got %s; want %s", got, test.want)
		}
	}
}

func TestFiles(t *testing.T) {
	var sources = []string{
		"package p; type T struct{}; func (T) m1() {}",
		"package p; func (T) m2() {}; var x interface{ m1(); m2() } = T{}",
		"package p; func (T) m3() {}; var y interface{ m1(); m2(); m3() } = T{}",
		"package p",
	}

	var conf Config
	fset := token.NewFileSet()
	pkg := NewPackage("p", "p")
	var info Info
	check := NewChecker(&conf, fset, pkg, &info)

	for _, src := range sources {
		if err := check.Files([]*ast.File{mustParse(fset, src)}); err != nil {
			t.Error(err)
		}
	}

	// check InitOrder is [x y]
	var vars []string
	for _, init := range info.InitOrder {
		for _, v := range init.Lhs {
			vars = append(vars, v.Name())
		}
	}
	if got, want := fmt.Sprint(vars), "[x y]"; got != want {
		t.Errorf("InitOrder == %s, want %s", got, want)
	}
}

type testImporter map[string]*Package

func (m testImporter) Import(path string) (*Package, error) {
	if pkg := m[path]; pkg != nil {
		return pkg, nil
	}
	return nil, fmt.Errorf("package %q not found", path)
}

func TestSelection(t *testing.T) {
	selections := make(map[*ast.SelectorExpr]*Selection)

	// We need a specific fileset in this test below for positions.
	// Cannot use typecheck helper.
	fset := token.NewFileSet()
	imports := make(testImporter)
	conf := Config{Importer: imports}
	makePkg := func(path, src string) {
		pkg, err := conf.Check(path, fset, []*ast.File{mustParse(fset, src)}, &Info{Selections: selections})
		if err != nil {
			t.Fatal(err)
		}
		imports[path] = pkg
	}

	const libSrc = `
package lib
type T float64
const C T = 3
var V T
func F() {}
func (T) M() {}
`
	const mainSrc = `
package main
import "lib"

type A struct {
	*B
	C
}

type B struct {
	b int
}

func (B) f(int)

type C struct {
	c int
}

type G[P any] struct {
	p P
}

func (G[P]) m(P) {}

var Inst G[int]

func (C) g()
func (*C) h()

func main() {
	// qualified identifiers
	var _ lib.T
	_ = lib.C
	_ = lib.F
	_ = lib.V
	_ = lib.T.M

	// fields
	_ = A{}.B
	_ = new(A).B

	_ = A{}.C
	_ = new(A).C

	_ = A{}.b
	_ = new(A).b

	_ = A{}.c
	_ = new(A).c

	_ = Inst.p
	_ = G[string]{}.p

	// methods
	_ = A{}.f
	_ = new(A).f
	_ = A{}.g
	_ = new(A).g
	_ = new(A).h

	_ = B{}.f
	_ = new(B).f

	_ = C{}.g
	_ = new(C).g
	_ = new(C).h
	_ = Inst.m

	// method expressions
	_ = A.f
	_ = (*A).f
	_ = B.f
	_ = (*B).f
	_ = G[string].m
}`

	wantOut := map[string][2]string{
		"lib.T.M": {"method expr (lib.T) M(lib.T)", ".[0]"},

		"A{}.B":    {"field (main.A) B *main.B", ".[0]"},
		"new(A).B": {"field (*main.A) B *main.B", "->[0]"},
		"A{}.C":    {"field (main.A) C main.C", ".[1]"},
		"new(A).C": {"field (*main.A) C main.C", "->[1]"},
		"A{}.b":    {"field (main.A) b int", "->[0 0]"},
		"new(A).b": {"field (*main.A) b int", "->[0 0]"},
		"A{}.c":    {"field (main.A) c int", ".[1 0]"},
		"new(A).c": {"field (*main.A) c int", "->[1 0]"},
		"Inst.p":   {"field (main.G[int]) p int", ".[0]"},

		"A{}.f":    {"method (main.A) f(int)", "->[0 0]"},
		"new(A).f": {"method (*main.A) f(int)", "->[0 0]"},
		"A{}.g":    {"method (main.A) g()", ".[1 0]"},
		"new(A).g": {"method (*main.A) g()", "->[1 0]"},
		"new(A).h": {"method (*main.A) h()", "->[1 1]"}, // TODO(gri) should this report .[1 1] ?
		"B{}.f":    {"method (main.B) f(int)", ".[0]"},
		"new(B).f": {"method (*main.B) f(int)", "->[0]"},
		"C{}.g":    {"method (main.C) g()", ".[0]"},
		"new(C).g": {"method (*main.C) g()", "->[0]"},
		"new(C).h": {"method (*main.C) h()", "->[1]"}, // TODO(gri) should this report .[1] ?
		"Inst.m":   {"method (main.G[int]) m(int)", ".[0]"},

		"A.f":           {"method expr (main.A) f(main.A, int)", "->[0 0]"},
		"(*A).f":        {"method expr (*main.A) f(*main.A, int)", "->[0 0]"},
		"B.f":           {"method expr (main.B) f(main.B, int)", ".[0]"},
		"(*B).f":        {"method expr (*main.B) f(*main.B, int)", "->[0]"},
		"G[string].m":   {"method expr (main.G[string]) m(main.G[string], string)", ".[0]"},
		"G[string]{}.p": {"field (main.G[string]) p string", ".[0]"},
	}

	makePkg("lib", libSrc)
	makePkg("main", mainSrc)

	for e, sel := range selections {
		_ = sel.String() // assertion: must not panic

		start := fset.Position(e.Pos()).Offset
		end := fset.Position(e.End()).Offset
		syntax := mainSrc[start:end] // (all SelectorExprs are in main, not lib)

		direct := "."
		if sel.Indirect() {
			direct = "->"
		}
		got := [2]string{
			sel.String(),
			fmt.Sprintf("%s%v", direct, sel.Index()),
		}
		want := wantOut[syntax]
		if want != got {
			t.Errorf("%s: got %q; want %q", syntax, got, want)
		}
		delete(wantOut, syntax)

		// We must explicitly assert properties of the
		// Signature's receiver since it doesn't participate
		// in Identical() or String().
		sig, _ := sel.Type().(*Signature)
		if sel.Kind() == MethodVal {
			got := sig.Recv().Type()
			want := sel.Recv()
			if !Identical(got, want) {
				t.Errorf("%s: Recv() = %s, want %s", syntax, got, want)
			}
		} else if sig != nil && sig.Recv() != nil {
			t.Errorf("%s: signature has receiver %s", sig, sig.Recv().Type())
		}
	}
	// Assert that all wantOut entries were used exactly once.
	for syntax := range wantOut {
		t.Errorf("no ast.Selection found with syntax %q", syntax)
	}
}

func TestIssue8518(t *testing.T) {
	fset := token.NewFileSet()
	imports := make(testImporter)
	conf := Config{
		Error:    func(err error) { t.Log(err) }, // don't exit after first error
		Importer: imports,
	}
	makePkg := func(path, src string) {
		imports[path], _ = conf.Check(path, fset, []*ast.File{mustParse(fset, src)}, nil) // errors logged via conf.Error
	}

	const libSrc = `
package a
import "missing"
const C1 = foo
const C2 = missing.C
`

	const mainSrc = `
package main
import "a"
var _ = a.C1
var _ = a.C2
`

	makePkg("a", libSrc)
	makePkg("main", mainSrc) // don't crash when type-checking this package
}

func TestIssue59603(t *testing.T) {
	fset := token.NewFileSet()
	imports := make(testImporter)
	conf := Config{
		Error:    func(err error) { t.Log(err) }, // don't exit after first error
		Importer: imports,
	}
	makePkg := func(path, src string) {
		imports[path], _ = conf.Check(path, fset, []*ast.File{mustParse(fset, src)}, nil) // errors logged via conf.Error
	}

	const libSrc = `
package a
const C = foo
`

	const mainSrc = `
package main
import "a"
const _ = a.C
`

	makePkg("a", libSrc)
	makePkg("main", mainSrc) // don't crash when type-checking this package
}

func TestLookupFieldOrMethodOnNil(t *testing.T) {
	// LookupFieldOrMethod on a nil type is expected to produce a run-time panic.
	defer func() {
		const want = "LookupFieldOrMethod on nil type"
		p := recover()
		if s, ok := p.(string); !ok || s != want {
			t.Fatalf("got %v, want %s", p, want)
		}
	}()
	LookupFieldOrMethod(nil, false, nil, "")
}

func TestLookupFieldOrMethod(t *testing.T) {
	// Test cases assume a lookup of the form a.f or x.f, where a stands for an
	// addressable value, and x for a non-addressable value (even though a variable
	// for ease of test case writing).
	//
	// Should be kept in sync with TestMethodSet.
	var tests = []struct {
		src      string
		found    bool
		index    []int
		indirect bool
	}{
		// field lookups
		{"var x T; type T struct{}", false, nil, false},
		{"var x T; type T struct{ f int }", true, []int{0}, false},
		{"var x T; type T struct{ a, b, f, c int }", true, []int{2}, false},

		// field lookups on a generic type
		{"var x T[int]; type T[P any] struct{}", false, nil, false},
		{"var x T[int]; type T[P any] struct{ f P }", true, []int{0}, false},
		{"var x T[int]; type T[P any] struct{ a, b, f, c P }", true, []int{2}, false},

		// method lookups
		{"var a T; type T struct{}; func (T) f() {}", true, []int{0}, false},
		{"var a *T; type T struct{}; func (T) f() {}", true, []int{0}, true},
		{"var a T; type T struct{}; func (*T) f() {}", true, []int{0}, false},
		{"var a *T; type T struct{}; func (*T) f() {}", true, []int{0}, true}, // TODO(gri) should this report indirect = false?

		// method lookups on a generic type
		{"var a T[int]; type T[P any] struct{}; func (T[P]) f() {}", true, []int{0}, false},
		{"var a *T[int]; type T[P any] struct{}; func (T[P]) f() {}", true, []int{0}, true},
		{"var a T[int]; type T[P any] struct{}; func (*T[P]) f() {}", true, []int{0}, false},
		{"var a *T[int]; type T[P any] struct{}; func (*T[P]) f() {}", true, []int{0}, true}, // TODO(gri) should this report indirect = false?

		// collisions
		{"type ( E1 struct{ f int }; E2 struct{ f int }; x struct{ E1; *E2 })", false, []int{1, 0}, false},
		{"type ( E1 struct{ f int }; E2 struct{}; x struct{ E1; *E2 }); func (E2) f() {}", false, []int{1, 0}, false},

		// collisions on a generic type
		{"type ( E1[P any] struct{ f P }; E2[P any] struct{ f P }; x struct{ E1[int]; *E2[int] })", false, []int{1, 0}, false},
		{"type ( E1[P any] struct{ f P }; E2[P any] struct{}; x struct{ E1[int]; *E2[int] }); func (E2[P]) f() {}", false, []int{1, 0}, false},

		// outside methodset
		// (*T).f method exists, but value of type T is not addressable
		{"var x T; type T struct{}; func (*T) f() {}", false, nil, true},

		// outside method set of a generic type
		{"var x T[int]; type T[P any] struct{}; func (*T[P]) f() {}", false, nil, true},

		// recursive generic types; see go.dev/issue/52715
		{"var a T[int]; type ( T[P any] struct { *N[P] }; N[P any] struct { *T[P] } ); func (N[P]) f() {}", true, []int{0, 0}, true},
		{"var a T[int]; type ( T[P any] struct { *N[P] }; N[P any] struct { *T[P] } ); func (T[P]) f() {}", true, []int{0}, false},
	}

	for _, test := range tests {
		pkg := mustTypecheck("package p;"+test.src, nil, nil)

		obj := pkg.Scope().Lookup("a")
		if obj == nil {
			if obj = pkg.Scope().Lookup("x"); obj == nil {
				t.Errorf("%s: incorrect test case - no object a or x", test.src)
				continue
			}
		}

		f, index, indirect := LookupFieldOrMethod(obj.Type(), obj.Name() == "a", pkg, "f")
		if (f != nil) != test.found {
			if f == nil {
				t.Errorf("%s: got no object; want one", test.src)
			} else {
				t.Errorf("%s: got object = %v; want none", test.src, f)
			}
		}
		if !sameSlice(index, test.index) {
			t.Errorf("%s: got index = %v; want %v", test.src, index, test.index)
		}
		if indirect != test.indirect {
			t.Errorf("%s: got indirect = %v; want %v", test.src, indirect, test.indirect)
		}
	}
}

// Test for go.dev/issue/52715
func TestLookupFieldOrMethod_RecursiveGeneric(t *testing.T) {
	const src = `
package pkg

type Tree[T any] struct {
	*Node[T]
}

func (*Tree[R]) N(r R) R { return r }

type Node[T any] struct {
	*Tree[T]
}

type Instance = *Tree[int]
`

	fset := token.NewFileSet()
	f := mustParse(fset, src)
	pkg := NewPackage("pkg", f.Name.Name)
	if err := NewChecker(nil, fset, pkg, nil).Files([]*ast.File{f}); err != nil {
		panic(err)
	}

	T := pkg.Scope().Lookup("Instance").Type()
	_, _, _ = LookupFieldOrMethod(T, false, pkg, "M") // verify that LookupFieldOrMethod terminates
}

func sameSlice(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, x := range a {
		if x != b[i] {
			return false
		}
	}
	return true
}

// TestScopeLookupParent ensures that (*Scope).LookupParent returns
// the correct result at various positions with the source.
func TestScopeLookupParent(t *testing.T) {
	fset := token.NewFileSet()
	imports := make(testImporter)
	conf := Config{Importer: imports}
	var info Info
	makePkg := func(path string, files ...*ast.File) {
		var err error
		imports[path], err = conf.Check(path, fset, files, &info)
		if err != nil {
			t.Fatal(err)
		}
	}

	makePkg("lib", mustParse(fset, "package lib; var X int"))
	// Each /*name=kind:line*/ comment makes the test look up the
	// name at that point and checks that it resolves to a decl of
	// the specified kind and line number.  "undef" means undefined.
	mainSrc := `
/*lib=pkgname:5*/ /*X=var:1*/ /*Pi=const:8*/ /*T=typename:9*/ /*Y=var:10*/ /*F=func:12*/
package main

import "lib"
import . "lib"

const Pi = 3.1415
type T struct{}
var Y, _ = lib.X, X

func F[T *U, U any](param1, param2 int) /*param1=undef*/ (res1 /*res1=undef*/, res2 int) /*param1=var:12*/ /*res1=var:12*/ /*U=typename:12*/ {
	const pi, e = 3.1415, /*pi=undef*/ 2.71828 /*pi=const:13*/ /*e=const:13*/
	type /*t=undef*/ t /*t=typename:14*/ *t
	print(Y) /*Y=var:10*/
	x, Y := Y, /*x=undef*/ /*Y=var:10*/ Pi /*x=var:16*/ /*Y=var:16*/ ; _ = x; _ = Y
	var F = /*F=func:12*/ F[*int, int] /*F=var:17*/ ; _ = F

	var a []int
	for i, x := range a /*i=undef*/ /*x=var:16*/ { _ = i; _ = x }

	var i interface{}
	switch y := i.(type) { /*y=undef*/
	case /*y=undef*/ int /*y=var:23*/ :
	case float32, /*y=undef*/ float64 /*y=var:23*/ :
	default /*y=var:23*/:
		println(y)
	}
	/*y=undef*/

        switch int := i.(type) {
        case /*int=typename:0*/ int /*int=var:31*/ :
        	println(int)
        default /*int=var:31*/ :
        }

	_ = param1
	_ = res1
	return
}
/*main=undef*/
`

	info.Uses = make(map[*ast.Ident]Object)
	f := mustParse(fset, mainSrc)
	makePkg("main", f)
	mainScope := imports["main"].Scope()
	rx := regexp.MustCompile(`^/\*(\w*)=([\w:]*)\*/$`)
	for _, group := range f.Comments {
		for _, comment := range group.List {
			// Parse the assertion in the comment.
			m := rx.FindStringSubmatch(comment.Text)
			if m == nil {
				t.Errorf("%s: bad comment: %s",
					fset.Position(comment.Pos()), comment.Text)
				continue
			}
			name, want := m[1], m[2]

			// Look up the name in the innermost enclosing scope.
			inner := mainScope.Innermost(comment.Pos())
			if inner == nil {
				t.Errorf("%s: at %s: can't find innermost scope",
					fset.Position(comment.Pos()), comment.Text)
				continue
			}
			got := "undef"
			if _, obj := inner.LookupParent(name, comment.Pos()); obj != nil {
				kind := strings.ToLower(strings.TrimPrefix(reflect.TypeOf(obj).String(), "*types."))
				got = fmt.Sprintf("%s:%d", kind, fset.Position(obj.Pos()).Line)
			}
			if got != want {
				t.Errorf("%s: at %s: %s resolved to %s, want %s",
					fset.Position(comment.Pos()), comment.Text, name, got, want)
			}
		}
	}

	// Check that for each referring identifier,
	// a lookup of its name on the innermost
	// enclosing scope returns the correct object.

	for id, wantObj := range info.Uses {
		inner := mainScope.Innermost(id.Pos())
		if inner == nil {
			t.Errorf("%s: can't find innermost scope enclosing %q",
				fset.Position(id.Pos()), id.Name)
			continue
		}

		// Exclude selectors and qualified identifiers---lexical
		// refs only.  (Ideally, we'd see if the AST parent is a
		// SelectorExpr, but that requires PathEnclosingInterval
		// from golang.org/x/tools/go/ast/astutil.)
		if id.Name == "X" {
			continue
		}

		_, gotObj := inner.LookupParent(id.Name, id.Pos())
		if gotObj != wantObj {
			// Print the scope tree of mainScope in case of error.
			var printScopeTree func(indent string, s *Scope)
			printScopeTree = func(indent string, s *Scope) {
				t.Logf("%sscope %s %v-%v = %v",
					indent,
					ScopeComment(s),
					s.Pos(),
					s.End(),
					s.Names())
				for i := range s.NumChildren() {
					printScopeTree(indent+"  ", s.Child(i))
				}
			}
			printScopeTree("", mainScope)

			t.Errorf("%s: Scope(%s).LookupParent(%s@%v) got %v, want %v [scopePos=%v]",
				fset.Position(id.Pos()),
				ScopeComment(inner),
				id.Name,
				id.Pos(),
				gotObj,
				wantObj,
				ObjectScopePos(wantObj))
			continue
		}
	}
}

// newDefined creates a new defined type named T with the given underlying type.
// Helper function for use with TestIncompleteInterfaces only.
func newDefined(underlying Type) *Named {
	tname := NewTypeName(nopos, nil, "T", nil)
	return NewNamed(tname, underlying, nil)
}

func TestConvertibleTo(t *testing.T) {
	for _, test := range []struct {
		v, t Type
		want bool
	}{
		{Typ[Int], Typ[Int], true},
		{Typ[Int], Typ[Float32], true},
		{Typ[Int], Typ[String], true},
		{newDefined(Typ[Int]), Typ[Int], true},
		{newDefined(new(Struct)), new(Struct), true},
		{newDefined(Typ[Int]), new(Struct), false},
		{Typ[UntypedInt], Typ[Int], true},
		{NewSlice(Typ[Int]), NewArray(Typ[Int], 10), true},
		{NewSlice(Typ[Int]), NewArray(Typ[Uint], 10), false},
		{NewSlice(Typ[Int]), NewPointer(NewArray(Typ[Int], 10)), true},
		{NewSlice(Typ[Int]), NewPointer(NewArray(Typ[Uint], 10)), false},
		// Untyped string values are not permitted by the spec, so the behavior below is undefined.
		{Typ[UntypedString], Typ[String], true},
	} {
		if got := ConvertibleTo(test.v, test.t); got != test.want {
			t.Errorf("ConvertibleTo(%v, %v) = %t, want %t", test.v, test.t, got, test.want)
		}
	}
}

func TestAssignableTo(t *testing.T) {
	for _, test := range []struct {
		v, t Type
		want bool
	}{
		{Typ[Int], Typ[Int], true},
		{Typ[Int], Typ[Float32], false},
		{newDefined(Typ[Int]), Typ[Int], false},
		{newDefined(new(Struct)), new(Struct), true},
		{Typ[UntypedBool], Typ[Bool], true},
		{Typ[UntypedString], Typ[Bool], false},
		// Neither untyped string nor untyped numeric assignments arise during
		// normal type checking, so the below behavior is technically undefined by
		// the spec.
		{Typ[UntypedString], Typ[String], true},
		{Typ[UntypedInt], Typ[Int], true},
	} {
		if got := AssignableTo(test.v, test.t); got != test.want {
			t.Errorf("AssignableTo(%v, %v) = %t, want %t", test.v, test.t, got, test.want)
		}
	}
}

func TestIdentical(t *testing.T) {
	// For each test, we compare the types of objects X and Y in the source.
	tests := []struct {
		src  string
		want bool
	}{
		// Basic types.
		{"var X int; var Y int", true},
		{"var X int; var Y string", false},

		// TODO: add more tests for complex types.

		// Named types.
		{"type X int; type Y int", false},

		// Aliases.
		{"type X = int; type Y = int", true},

		// Functions.
		{`func X(int) string { return "" }; func Y(int) string { return "" }`, true},
		{`func X() string { return "" }; func Y(int) string { return "" }`, false},
		{`func X(int) string { return "" }; func Y(int) {}`, false},

		// Generic functions. Type parameters should be considered identical modulo
		// renaming. See also go.dev/issue/49722.
		{`func X[P ~int](){}; func Y[Q ~int]() {}`, true},
		{`func X[P1 any, P2 ~*P1](){}; func Y[Q1 any, Q2 ~*Q1]() {}`, true},
		{`func X[P1 any, P2 ~[]P1](){}; func Y[Q1 any, Q2 ~*Q1]() {}`, false},
		{`func X[P ~int](P){}; func Y[Q ~int](Q) {}`, true},
		{`func X[P ~string](P){}; func Y[Q ~int](Q) {}`, false},
		{`func X[P ~int]([]P){}; func Y[Q ~int]([]Q) {}`, true},
	}

	for _, test := range tests {
		pkg := mustTypecheck("package p;"+test.src, nil, nil)
		X := pkg.Scope().Lookup("X")
		Y := pkg.Scope().Lookup("Y")
		if X == nil || Y == nil {
			t.Fatal("test must declare both X and Y")
		}
		if got := Identical(X.Type(), Y.Type()); got != test.want {
			t.Errorf("Identical(%s, %s) = %t, want %t", X.Type(), Y.Type(), got, test.want)
		}
	}
}

func TestIdentical_issue15173(t *testing.T) {
	// Identical should allow nil arguments and be symmetric.
	for _, test := range []struct {
		x, y Type
		want bool
	}{
		{Typ[Int], Typ[Int], true},
		{Typ[Int], nil, false},
		{nil, Typ[Int], false},
		{nil, nil, true},
	} {
		if got := Identical(test.x, test.y); got != test.want {
			t.Errorf("Identical(%v, %v) = %t", test.x, test.y, got)
		}
	}
}

func TestIdenticalUnions(t *testing.T) {
	tname := NewTypeName(nopos, nil, "myInt", nil)
	myInt := NewNamed(tname, Typ[Int], nil)
	tmap := map[string]*Term{
		"int":     NewTerm(false, Typ[Int]),
		"~int":    NewTerm(true, Typ[Int]),
		"string":  NewTerm(false, Typ[String]),
		"~string": NewTerm(true, Typ[String]),
		"myInt":   NewTerm(false, myInt),
	}
	makeUnion := func(s string) *Union {
		parts := strings.Split(s, "|")
		var terms []*Term
		for _, p := range parts {
			term := tmap[p]
			if term == nil {
				t.Fatalf("missing term %q", p)
			}
			terms = append(terms, term)
		}
		return NewUnion(terms)
	}
	for _, test := range []struct {
		x, y string
		want bool
	}{
		// These tests are just sanity checks. The tests for type sets and
		// interfaces provide much more test coverage.
		{"int|~int", "~int", true},
		{"myInt|~int", "~int", true},
		{"int|string", "string|int", true},
		{"int|int|string", "string|int", true},
		{"myInt|string", "int|string", false},
	} {
		x := makeUnion(test.x)
		y := makeUnion(test.y)
		if got := Identical(x, y); got != test.want {
			t.Errorf("Identical(%v, %v) = %t", test.x, test.y, got)
		}
	}
}

func TestIssue61737(t *testing.T) {
	// This test verifies that it is possible to construct invalid interfaces
	// containing duplicate methods using the go/types API.
	//
	// It must be possible for importers to construct such invalid interfaces.
	// Previously, this panicked.

	sig1 := NewSignatureType(nil, nil, nil, NewTuple(NewParam(nopos, nil, "", Typ[Int])), nil, false)
	sig2 := NewSignatureType(nil, nil, nil, NewTuple(NewParam(nopos, nil, "", Typ[String])), nil, false)

	methods := []*Func{
		NewFunc(nopos, nil, "M", sig1),
		NewFunc(nopos, nil, "M", sig2),
	}

	embeddedMethods := []*Func{
		NewFunc(nopos, nil, "M", sig2),
	}
	embedded := NewInterfaceType(embeddedMethods, nil)
	iface := NewInterfaceType(methods, []Type{embedded})
	iface.Complete()
}

func TestNewAlias_Issue65455(t *testing.T) {
	obj := NewTypeName(nopos, nil, "A", nil)
	alias := NewAlias(obj, Typ[Int])
	alias.Underlying() // must not panic
}

func TestIssue15305(t *testing.T) {
	const src = "package p; func f() int16; var _ = f(undef)"
	fset := token.NewFileSet()
	f := mustParse(fset, src)
	conf := Config{
		Error: func(err error) {}, // allow errors
	}
	info := &Info{
		Types: make(map[ast.Expr]TypeAndValue),
	}
	conf.Check("p", fset, []*ast.File{f}, info) // ignore result
	for e, tv := range info.Types {
		if _, ok := e.(*ast.CallExpr); ok {
			if tv.Type != Typ[Int16] {
				t.Errorf("CallExpr has type %v, want int16", tv.Type)
			}
			return
		}
	}
	t.Errorf("CallExpr has no type")
}

// TestCompositeLitTypes verifies that Info.Types registers the correct
// types for composite literal expressions and composite literal type
// expressions.
func TestCompositeLitTypes(t *testing.T) {
	for i, test := range []struct {
		lit, typ string
	}{
		{`[16]byte{}`, `[16]byte`},
		{`[...]byte{}`, `[0]byte`},                // test for go.dev/issue/14092
		{`[...]int{1, 2, 3}`, `[3]int`},           // test for go.dev/issue/14092
		{`[...]int{90: 0, 98: 1, 2}`, `[100]int`}, // test for go.dev/issue/14092
		{`[]int{}`, `[]int`},
		{`map[string]bool{"foo": true}`, `map[string]bool`},
		{`struct{}{}`, `struct{}`},
		{`struct{x, y int; z complex128}{}`, `struct{x int; y int; z complex128}`},
	} {
		fset := token.NewFileSet()
		f := mustParse(fset, fmt.Sprintf("package p%d; var _ = %s", i, test.lit))
		types := make(map[ast.Expr]TypeAndValue)
		if _, err := new(Config).Check("p", fset, []*ast.File{f}, &Info{Types: types}); err != nil {
			t.Fatalf("%s: %v", test.lit, err)
		}

		cmptype := func(x ast.Expr, want string) {
			tv, ok := types[x]
			if !ok {
				t.Errorf("%s: no Types entry found", test.lit)
				return
			}
			if tv.Type == nil {
				t.Errorf("%s: type is nil", test.lit)
				return
			}
			if got := tv.Type.String(); got != want {
				t.Errorf("%s: got %v, want %s", test.lit, got, want)
			}
		}

		// test type of composite literal expression
		rhs := f.Decls[0].(*ast.GenDecl).Specs[0].(*ast.ValueSpec).Values[0]
		cmptype(rhs, test.typ)

		// test type of composite literal type expression
		cmptype(rhs.(*ast.CompositeLit).Type, test.typ)
	}
}

// TestObjectParents verifies that objects have parent scopes or not
// as specified by the Object interface.
func TestObjectParents(t *testing.T) {
	const src = `
package p

const C = 0

type T1 struct {
	a, b int
	T2
}

type T2 interface {
	im1()
	im2()
}

func (T1) m1() {}
func (*T1) m2() {}

func f(x int) { y := x; print(y) }
`

	fset := token.NewFileSet()
	f := mustParse(fset, src)

	info := &Info{
		Defs: make(map[*ast.Ident]Object),
	}
	if _, err := new(Config).Check("p", fset, []*ast.File{f}, info); err != nil {
		t.Fatal(err)
	}

	for ident, obj := range info.Defs {
		if obj == nil {
			// only package names and implicit vars have a nil object
			// (in this test we only need to handle the package name)
			if ident.Name != "p" {
				t.Errorf("%v has nil object", ident)
			}
			continue
		}

		// struct fields, type-associated and interface methods
		// have no parent scope
		wantParent := true
		switch obj := obj.(type) {
		case *Var:
			if obj.IsField() {
				wantParent = false
			}
		case *Func:
			if obj.Type().(*Signature).Recv() != nil { // method
				wantParent = false
			}
		}

		gotParent := obj.Parent() != nil
		switch {
		case gotParent && !wantParent:
			t.Errorf("%v: want no parent, got %s", ident, obj.Parent())
		case !gotParent && wantParent:
			t.Errorf("%v: no parent found", ident)
		}
	}
}

// TestFailedImport tests that we don't get follow-on errors
// elsewhere in a package due to failing to import a package.
func TestFailedImport(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const src = `
package p

import foo "go/types/thisdirectorymustnotexistotherwisethistestmayfail/foo" // should only see an error here

const c = foo.C
type T = foo.T
var v T = c
func f(x T) T { return foo.F(x) }
`
	fset := token.NewFileSet()
	f := mustParse(fset, src)
	files := []*ast.File{f}

	// type-check using all possible importers
	for _, compiler := range []string{"gc", "gccgo", "source"} {
		errcount := 0
		conf := Config{
			Error: func(err error) {
				// we should only see the import error
				if errcount > 0 || !strings.Contains(err.Error(), "could not import") {
					t.Errorf("for %s importer, got unexpected error: %v", compiler, err)
				}
				errcount++
			},
			Importer: importer.For(compiler, nil),
		}

		info := &Info{
			Uses: make(map[*ast.Ident]Object),
		}
		pkg, _ := conf.Check("p", fset, files, info)
		if pkg == nil {
			t.Errorf("for %s importer, type-checking failed to return a package", compiler)
			continue
		}

		imports := pkg.Imports()
		if len(imports) != 1 {
			t.Errorf("for %s importer, got %d imports, want 1", compiler, len(imports))
			continue
		}
		imp := imports[0]
		if imp.Name() != "foo" {
			t.Errorf(`for %s importer, got %q, want "foo"`, compiler, imp.Name())
			continue
		}

		// verify that all uses of foo refer to the imported package foo (imp)
		for ident, obj := range info.Uses {
			if ident.Name == "foo" {
				if obj, ok := obj.(*PkgName); ok {
					if obj.Imported() != imp {
						t.Errorf("%s resolved to %v; want %v", ident, obj.Imported(), imp)
					}
				} else {
					t.Errorf("%s resolved to %v; want package name", ident, obj)
				}
			}
		}
	}
}

func TestInstantiate(t *testing.T) {
	// eventually we like more tests but this is a start
	const src = "package p; type T[P any] *T[P]"
	pkg := mustTypecheck(src, nil, nil)

	// type T should have one type parameter
	T := pkg.Scope().Lookup("T").Type().(*Named)
	if n := T.TypeParams().Len(); n != 1 {
		t.Fatalf("expected 1 type parameter; found %d", n)
	}

	// instantiation should succeed (no endless recursion)
	// even with a nil *Checker
	res, err := Instantiate(nil, T, []Type{Typ[Int]}, false)
	if err != nil {
		t.Fatal(err)
	}

	// instantiated type should point to itself
	if p := res.Underlying().(*Pointer).Elem(); p != res {
		t.Fatalf("unexpected result type: %s points to %s", res, p)
	}
}

func TestInstantiateConcurrent(t *testing.T) {
	const src = `package p

type I[P any] interface {
	m(P)
	n() P
}

type J = I[int]

type Nested[P any] *interface{b(P)}

type K = Nested[string]
`
	pkg := mustTypecheck(src, nil, nil)

	insts := []*Interface{
		pkg.Scope().Lookup("J").Type().Underlying().(*Interface),
		pkg.Scope().Lookup("K").Type().Underlying().(*Pointer).Elem().(*Interface),
	}

	// Use the interface instances concurrently.
	for _, inst := range insts {
		var (
			counts  [2]int      // method counts
			methods [2][]string // method strings
		)
		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			i := i
			wg.Add(1)
			go func() {
				defer wg.Done()

				counts[i] = inst.NumMethods()
				for mi := 0; mi < counts[i]; mi++ {
					methods[i] = append(methods[i], inst.Method(mi).String())
				}
			}()
		}
		wg.Wait()

		if counts[0] != counts[1] {
			t.Errorf("mismatching method counts for %s: %d vs %d", inst, counts[0], counts[1])
			continue
		}
		for i := 0; i < counts[0]; i++ {
			if m0, m1 := methods[0][i], methods[1][i]; m0 != m1 {
				t.Errorf("mismatching methods for %s: %s vs %s", inst, m0, m1)
			}
		}
	}
}

func TestInstantiateErrors(t *testing.T) {
	tests := []struct {
		src    string // by convention, T must be the type being instantiated
		targs  []Type
		wantAt int // -1 indicates no error
	}{
		{"type T[P interface{~string}] int", []Type{Typ[Int]}, 0},
		{"type T[P1 interface{int}, P2 interface{~string}] int", []Type{Typ[Int], Typ[Int]}, 1},
		{"type T[P1 any, P2 interface{~[]P1}] int", []Type{Typ[Int], NewSlice(Typ[String])}, 1},
		{"type T[P1 interface{~[]P2}, P2 any] int", []Type{NewSlice(Typ[String]), Typ[Int]}, 0},
	}

	for _, test := range tests {
		src := "package p; " + test.src
		pkg := mustTypecheck(src, nil, nil)

		T := pkg.Scope().Lookup("T").Type().(*Named)

		_, err := Instantiate(nil, T, test.targs, true)
		if err == nil {
			t.Fatalf("Instantiate(%v, %v) returned nil error, want non-nil", T, test.targs)
		}

		var argErr *ArgumentError
		if !errors.As(err, &argErr) {
			t.Fatalf("Instantiate(%v, %v): error is not an *ArgumentError", T, test.targs)
		}

		if argErr.Index != test.wantAt {
			t.Errorf("Instantiate(%v, %v): error at index %d, want index %d", T, test.targs, argErr.Index, test.wantAt)
		}
	}
}

func TestArgumentErrorUnwrapping(t *testing.T) {
	var err error = &ArgumentError{
		Index: 1,
		Err:   Error{Msg: "test"},
	}
	var e Error
	if !errors.As(err, &e) {
		t.Fatalf("error %v does not wrap types.Error", err)
	}
	if e.Msg != "test" {
		t.Errorf("e.Msg = %q, want %q", e.Msg, "test")
	}
}

func TestInstanceIdentity(t *testing.T) {
	imports := make(testImporter)
	conf := Config{Importer: imports}
	makePkg := func(src string) {
		fset := token.NewFileSet()
		f := mustParse(fset, src)
		name := f.Name.Name
		pkg, err := conf.Check(name, fset, []*ast.File{f}, nil)
		if err != nil {
			t.Fatal(err)
		}
		imports[name] = pkg
	}
	makePkg(`package lib; type T[P any] struct{}`)
	makePkg(`package a; import "lib"; var A lib.T[int]`)
	makePkg(`package b; import "lib"; var B lib.T[int]`)
	a := imports["a"].Scope().Lookup("A")
	b := imports["b"].Scope().Lookup("B")
	if !Identical(a.Type(), b.Type()) {
		t.Errorf("mismatching types: a.A: %s, b.B: %s", a.Type(), b.Type())
	}
}

// TestInstantiatedObjects verifies properties of instantiated objects.
func TestInstantiatedObjects(t *testing.T) {
	const src = `
package p

type T[P any] struct {
	field P
}

func (recv *T[Q]) concreteMethod(mParam Q) (mResult Q) { return }

type FT[P any] func(ftParam P) (ftResult P)

func F[P any](fParam P) (fResult P){ return }

type I[P any] interface {
	interfaceMethod(P)
}

type R[P any] T[P]

func (R[P]) m() {} // having a method triggers expansion of R

var (
	t T[int]
	ft FT[int]
	f = F[int]
	i I[int]
)

func fn() {
	var r R[int]
	_ = r
}
`
	info := &Info{
		Defs: make(map[*ast.Ident]Object),
	}
	fset := token.NewFileSet()
	f := mustParse(fset, src)
	conf := Config{}
	pkg, err := conf.Check(f.Name.Name, fset, []*ast.File{f}, info)
	if err != nil {
		t.Fatal(err)
	}

	lookup := func(name string) Type { return pkg.Scope().Lookup(name).Type() }
	fnScope := pkg.Scope().Lookup("fn").(*Func).Scope()

	tests := []struct {
		name string
		obj  Object
	}{
		// Struct fields
		{"field", lookup("t").Underlying().(*Struct).Field(0)},
		{"field", fnScope.Lookup("r").Type().Underlying().(*Struct).Field(0)},

		// Methods and method fields
		{"concreteMethod", lookup("t").(*Named).Method(0)},
		{"recv", lookup("t").(*Named).Method(0).Type().(*Signature).Recv()},
		{"mParam", lookup("t").(*Named).Method(0).Type().(*Signature).Params().At(0)},
		{"mResult", lookup("t").(*Named).Method(0).Type().(*Signature).Results().At(0)},

		// Interface methods
		{"interfaceMethod", lookup("i").Underlying().(*Interface).Method(0)},

		// Function type fields
		{"ftParam", lookup("ft").Underlying().(*Signature).Params().At(0)},
		{"ftResult", lookup("ft").Underlying().(*Signature).Results().At(0)},

		// Function fields
		{"fParam", lookup("f").(*Signature).Params().At(0)},
		{"fResult", lookup("f").(*Signature).Results().At(0)},
	}

	// Collect all identifiers by name.
	idents := make(map[string][]*ast.Ident)
	ast.Inspect(f, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			idents[id.Name] = append(idents[id.Name], id)
		}
		return true
	})

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			if got := len(idents[test.name]); got != 1 {
				t.Fatalf("found %d identifiers named %s, want 1", got, test.name)
			}
			ident := idents[test.name][0]
			def := info.Defs[ident]
			if def == test.obj {
				t.Fatalf("info.Defs[%s] contains the test object", test.name)
			}
			if orig := originObject(test.obj); def != orig {
				t.Errorf("info.Defs[%s] does not match obj.Origin()", test.name)
			}
			if def.Pkg() != test.obj.Pkg() {
				t.Errorf("Pkg() = %v, want %v", def.Pkg(), test.obj.Pkg())
			}
			if def.Name() != test.obj.Name() {
				t.Errorf("Name() = %v, want %v", def.Name(), test.obj.Name())
			}
			if def.Pos() != test.obj.Pos() {
				t.Errorf("Pos() = %v, want %v", def.Pos(), test.obj.Pos())
			}
			if def.Parent() != test.obj.Parent() {
				t.Fatalf("Parent() = %v, want %v", def.Parent(), test.obj.Parent())
			}
			if def.Exported() != test.obj.Exported() {
				t.Fatalf("Exported() = %v, want %v", def.Exported(), test.obj.Exported())
			}
			if def.Id() != test.obj.Id() {
				t.Fatalf("Id() = %v, want %v", def.Id(), test.obj.Id())
			}
			// String and Type are expected to differ.
		})
	}
}

func originObject(obj Object) Object {
	switch obj := obj.(type) {
	case *Var:
		return obj.Origin()
	case *Func:
		return obj.Origin()
	}
	return obj
}

func TestImplements(t *testing.T) {
	const src = `
package p

type EmptyIface interface{}

type I interface {
	m()
}

type C interface {
	m()
	~int
}

type Integer interface{
	int8 | int16 | int32 | int64
}

type EmptyTypeSet interface{
	Integer
	~string
}

type N1 int
func (N1) m() {}

type N2 int
func (*N2) m() {}

type N3 int
func (N3) m(int) {}

type N4 string
func (N4) m()

type Bad Bad // invalid type
`

	fset := token.NewFileSet()
	f := mustParse(fset, src)
	conf := Config{Error: func(error) {}}
	pkg, _ := conf.Check(f.Name.Name, fset, []*ast.File{f}, nil)

	lookup := func(tname string) Type { return pkg.Scope().Lookup(tname).Type() }
	var (
		EmptyIface   = lookup("EmptyIface").Underlying().(*Interface)
		I            = lookup("I").(*Named)
		II           = I.Underlying().(*Interface)
		C            = lookup("C").(*Named)
		CI           = C.Underlying().(*Interface)
		Integer      = lookup("Integer").Underlying().(*Interface)
		EmptyTypeSet = lookup("EmptyTypeSet").Underlying().(*Interface)
		N1           = lookup("N1")
		N1p          = NewPointer(N1)
		N2           = lookup("N2")
		N2p          = NewPointer(N2)
		N3           = lookup("N3")
		N4           = lookup("N4")
		Bad          = lookup("Bad")
	)

	tests := []struct {
		V    Type
		T    *Interface
		want bool
	}{
		{I, II, true},
		{I, CI, false},
		{C, II, true},
		{C, CI, true},
		{Typ[Int8], Integer, true},
		{Typ[Int64], Integer, true},
		{Typ[String], Integer, false},
		{EmptyTypeSet, II, true},
		{EmptyTypeSet, EmptyTypeSet, true},
		{Typ[Int], EmptyTypeSet, false},
		{N1, II, true},
		{N1, CI, true},
		{N1p, II, true},
		{N1p, CI, false},
		{N2, II, false},
		{N2, CI, false},
		{N2p, II, true},
		{N2p, CI, false},
		{N3, II, false},
		{N3, CI, false},
		{N4, II, true},
		{N4, CI, false},
		{Bad, II, false},
		{Bad, CI, false},
		{Bad, EmptyIface, true},
	}

	for _, test := range tests {
		if got := Implements(test.V, test.T); got != test.want {
			t.Errorf("Implements(%s, %s) = %t, want %t", test.V, test.T, got, test.want)
		}

		// The type assertion x.(T) is valid if T is an interface or if T implements the type of x.
		// The assertion is never valid if T is a bad type.
		V := test.T
		T := test.V
		want := false
		if _, ok := T.Underlying().(*Interface); (ok || Implements(T, V)) && T != Bad {
			want = true
		}
		if got := AssertableTo(V, T); got != want {
			t.Errorf("AssertableTo(%s, %s) = %t, want %t", V, T, got, want)
		}
	}
}

func TestMissingMethodAlternative(t *testing.T) {
	const src = `
package p
type T interface {
	m()
}

type V0 struct{}
func (V0) m() {}

type V1 struct{}

type V2 struct{}
func (V2) m() int

type V3 struct{}
func (*V3) m()

type V4 struct{}
func (V4) M()
`

	pkg := mustTypecheck(src, nil, nil)

	T := pkg.Scope().Lookup("T").Type().Underlying().(*Interface)
	lookup := func(name string) (*Func, bool) {
		return MissingMethod(pkg.Scope().Lookup(name).Type(), T, true)
	}

	// V0 has method m with correct signature. Should not report wrongType.
	method, wrongType := lookup("V0")
	if method != nil || wrongType {
		t.Fatalf("V0: got method = %v, wrongType = %v", method, wrongType)
	}

	checkMissingMethod := func(tname string, reportWrongType bool) {
		method, wrongType := lookup(tname)
		if method == nil || method.Name() != "m" || wrongType != reportWrongType {
			t.Fatalf("%s: got method = %v, wrongType = %v", tname, method, wrongType)
		}
	}

	// V1 has no method m. Should not report wrongType.
	checkMissingMethod("V1", false)

	// V2 has method m with wrong signature type (ignoring receiver). Should report wrongType.
	checkMissingMethod("V2", true)

	// V3 has no method m but it exists on *V3. Should report wrongType.
	checkMissingMethod("V3", true)

	// V4 has no method m but has M. Should not report wrongType.
	checkMissingMethod("V4", false)
}

func TestErrorURL(t *testing.T) {
	var conf Config
	*stringFieldAddr(&conf, "_ErrorURL") = " [go.dev/e/%s]"

	// test case for a one-line error
	const src1 = `
package p
var _ T
`
	_, err := typecheck(src1, &conf, nil)
	if err == nil || !strings.HasSuffix(err.Error(), " [go.dev/e/UndeclaredName]") {
		t.Errorf("src1: unexpected error: got %v", err)
	}

	// test case for a multi-line error
	const src2 = `
package p
func f() int { return 0 }
var _ = f(1, 2)
`
	_, err = typecheck(src2, &conf, nil)
	if err == nil || !strings.Contains(err.Error(), " [go.dev/e/WrongArgCount]\n") {
		t.Errorf("src1: unexpected error: got %v", err)
	}
}

func TestModuleVersion(t *testing.T) {
	// version go1.dd must be able to typecheck go1.dd.0, go1.dd.1, etc.
	goversion := fmt.Sprintf("go1.%d", goversion.Version)
	for _, v := range []string{
		goversion,
		goversion + ".0",
		goversion + ".1",
		goversion + ".rc",
	} {
		conf := Config{GoVersion: v}
		pkg := mustTypecheck("package p", &conf, nil)
		if pkg.GoVersion() != conf.GoVersion {
			t.Errorf("got %s; want %s", pkg.GoVersion(), conf.GoVersion)
		}
	}
}

func TestFileVersions(t *testing.T) {
	for _, test := range []struct {
		goVersion   string
		fileVersion string
		wantVersion string
	}{
		{"", "", ""},                   // no versions specified
		{"go1.19", "", "go1.19"},       // module version specified
		{"", "go1.20", ""},             // file upgrade ignored
		{"go1.19", "go1.20", "go1.20"}, // file upgrade permitted
		{"go1.20", "go1.19", "go1.20"}, // file downgrade not permitted
		{"go1.21", "go1.19", "go1.19"}, // file downgrade permitted (module version is >= go1.21)

		// versions containing release numbers
		// (file versions containing release numbers are considered invalid)
		{"go1.19.0", "", "go1.19.0"},         // no file version specified
		{"go1.20", "go1.20.1", "go1.20"},     // file upgrade ignored
		{"go1.20.1", "go1.20", "go1.20.1"},   // file upgrade ignored
		{"go1.20.1", "go1.21", "go1.21"},     // file upgrade permitted
		{"go1.20.1", "go1.19", "go1.20.1"},   // file downgrade not permitted
		{"go1.21.1", "go1.19.1", "go1.21.1"}, // file downgrade not permitted (invalid file version)
		{"go1.21.1", "go1.19", "go1.19"},     // file downgrade permitted (module version is >= go1.21)
	} {
		var src string
		if test.fileVersion != "" {
			src = "//go:build " + test.fileVersion + "\n"
		}
		src += "package p"

		conf := Config{GoVersion: test.goVersion}
		versions := make(map[*ast.File]string)
		var info Info
		info.FileVersions = versions
		mustTypecheck(src, &conf, &info)

		n := 0
		for _, v := range versions {
			want := test.wantVersion
			if v != want {
				t.Errorf("%q: unexpected file version: got %q, want %q", src, v, want)
			}
			n++
		}
		if n != 1 {
			t.Errorf("%q: incorrect number of map entries: got %d", src, n)
		}
	}
}

// TestTooNew ensures that "too new" errors are emitted when the file
// or module is tagged with a newer version of Go than this go/types.
func TestTooNew(t *testing.T) {
	for _, test := range []struct {
		goVersion   string // package's Go version (as if derived from go.mod file)
		fileVersion string // file's Go version (becomes a build tag)
		wantErr     string // expected substring of concatenation of all errors
	}{
		{"go1.98", "", "package requires newer Go version go1.98"},
		{"", "go1.99", "p:2:9: file requires newer Go version go1.99"},
		{"go1.98", "go1.99", "package requires newer Go version go1.98"}, // (two
		{"go1.98", "go1.99", "file requires newer Go version go1.99"},    // errors)
	} {
		var src string
		if test.fileVersion != "" {
			src = "//go:build " + test.fileVersion + "\n"
		}
		src += "package p; func f()"

		var errs []error
		conf := Config{
			GoVersion: test.goVersion,
			Error:     func(err error) { errs = append(errs, err) },
		}
		info := &Info{Defs: make(map[*ast.Ident]Object)}
		typecheck(src, &conf, info)
		got := fmt.Sprint(errs)
		if !strings.Contains(got, test.wantErr) {
			t.Errorf("%q: unexpected error: got %q, want substring %q",
				src, got, test.wantErr)
		}

		// Assert that declarations were type checked nonetheless.
		var gotObjs []string
		for id, obj := range info.Defs {
			if obj != nil {
				objStr := strings.ReplaceAll(fmt.Sprintf("%s:%T", id.Name, obj), "types2", "types")
				gotObjs = append(gotObjs, objStr)
			}
		}
		wantObjs := "f:*types.Func"
		if !strings.Contains(fmt.Sprint(gotObjs), wantObjs) {
			t.Errorf("%q: got %s, want substring %q",
				src, gotObjs, wantObjs)
		}
	}
}
