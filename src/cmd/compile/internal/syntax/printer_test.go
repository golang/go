// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
)

func TestPrint(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	ast, _ := ParseFile(*src_, func(err error) { t.Error(err) }, nil, 0)

	if ast != nil {
		Fprint(testOut(), ast, LineForm)
		fmt.Println()
	}
}

type shortBuffer struct {
	buf []byte
}

func (w *shortBuffer) Write(data []byte) (n int, err error) {
	w.buf = append(w.buf, data...)
	n = len(data)
	if len(w.buf) > 10 {
		err = io.ErrShortBuffer
	}
	return
}

func TestPrintError(t *testing.T) {
	const src = "package p; var x int"
	ast, err := Parse(nil, strings.NewReader(src), nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}

	var buf shortBuffer
	_, err = Fprint(&buf, ast, 0)
	if err == nil || !errors.Is(err, io.ErrShortBuffer) {
		t.Errorf("got err = %s, want %s", err, io.ErrShortBuffer)
	}
}

var stringTests = [][2]string{
	dup("package p"),
	dup("package p; type _ int; type T1 = struct{}; type ( _ *struct{}; T2 = float32 )"),

	// generic type declarations (given type separated with blank from LHS)
	dup("package p; type _[T any] struct{}"),
	dup("package p; type _[A, B, C interface{m()}] struct{}"),
	dup("package p; type _[T any, A, B, C interface{m()}, X, Y, Z interface{~int}] struct{}"),

	dup("package p; type _[P *struct{}] struct{}"),
	dup("package p; type _[P *T,] struct{}"),
	dup("package p; type _[P *T, _ any] struct{}"),
	{"package p; type _[P (*T),] struct{}", "package p; type _[P *T,] struct{}"},
	{"package p; type _[P (*T), _ any] struct{}", "package p; type _[P *T, _ any] struct{}"},
	{"package p; type _[P (T),] struct{}", "package p; type _[P T] struct{}"},
	{"package p; type _[P (T), _ any] struct{}", "package p; type _[P T, _ any] struct{}"},

	{"package p; type _[P (*struct{})] struct{}", "package p; type _[P *struct{}] struct{}"},
	{"package p; type _[P ([]int)] struct{}", "package p; type _[P []int] struct{}"},
	{"package p; type _[P ([]int) | int] struct{}", "package p; type _[P []int | int] struct{}"},

	// a type literal in an |-expression indicates a type parameter list (blank after type parameter list and type)
	dup("package p; type _[P *[]int] struct{}"),
	dup("package p; type _[P T | T] struct{}"),
	dup("package p; type _[P T | T | T | T] struct{}"),
	dup("package p; type _[P *T | T, Q T] struct{}"),
	dup("package p; type _[P *[]T | T] struct{}"),
	dup("package p; type _[P *T | T | T | T | ~T] struct{}"),
	dup("package p; type _[P *T | T | T | ~T | T] struct{}"),
	dup("package p; type _[P *T | T | struct{} | T] struct{}"),
	dup("package p; type _[P <-chan int] struct{}"),
	dup("package p; type _[P *T | struct{} | T] struct{}"),

	// a trailing comma always indicates a (possibly invalid) type parameter list (blank after type parameter list and type)
	dup("package p; type _[P *T,] struct{}"),
	dup("package p; type _[P *T | T,] struct{}"),
	dup("package p; type _[P *T | <-T | T,] struct{}"),

	// slice/array type declarations (no blank between array length and element type)
	dup("package p; type _ []byte"),
	dup("package p; type _ [n]byte"),
	dup("package p; type _ [P(T)]byte"),
	dup("package p; type _ [P((T))]byte"),
	dup("package p; type _ [P * *T]byte"),
	dup("package p; type _ [P * T]byte"),
	dup("package p; type _ [P(*T)]byte"),
	dup("package p; type _ [P(**T)]byte"),
	dup("package p; type _ [P * T - T]byte"),
	dup("package p; type _ [P * T - T]byte"),
	dup("package p; type _ [P * T | T]byte"),
	dup("package p; type _ [P * T | <-T | T]byte"),

	// generic function declarations
	dup("package p; func _[T any]()"),
	dup("package p; func _[A, B, C interface{m()}]()"),
	dup("package p; func _[T any, A, B, C interface{m()}, X, Y, Z interface{~int}]()"),

	// generic functions with elided interfaces in type constraints
	dup("package p; func _[P *T]() {}"),
	dup("package p; func _[P *T | T | T | T | ~T]() {}"),
	dup("package p; func _[P *T | T | struct{} | T]() {}"),
	dup("package p; func _[P ~int, Q int | string]() {}"),
	dup("package p; func _[P struct{f int}, Q *P]() {}"),

	// methods with generic receiver types
	dup("package p; func (R[T]) _()"),
	dup("package p; func (*R[A, B, C]) _()"),
	dup("package p; func (_ *R[A, B, C]) _()"),

	// channels
	dup("package p; type _ chan chan int"),
	dup("package p; type _ chan (<-chan int)"),
	dup("package p; type _ chan chan<- int"),

	dup("package p; type _ <-chan chan int"),
	dup("package p; type _ <-chan <-chan int"),
	dup("package p; type _ <-chan chan<- int"),

	dup("package p; type _ chan<- chan int"),
	dup("package p; type _ chan<- <-chan int"),
	dup("package p; type _ chan<- chan<- int"),

	// TODO(gri) expand
}

func TestPrintString(t *testing.T) {
	for _, test := range stringTests {
		ast, err := Parse(nil, strings.NewReader(test[0]), nil, nil, 0)
		if err != nil {
			t.Error(err)
			continue
		}
		if got := String(ast); got != test[1] {
			t.Errorf("%q: got %q", test[1], got)
		}
	}
}

func testOut() io.Writer {
	if testing.Verbose() {
		return os.Stdout
	}
	return io.Discard
}

func dup(s string) [2]string { return [2]string{s, s} }

var exprTests = [][2]string{
	// basic type literals
	dup("x"),
	dup("true"),
	dup("42"),
	dup("3.1415"),
	dup("2.71828i"),
	dup(`'a'`),
	dup(`"foo"`),
	dup("`bar`"),
	dup("any"),

	// func and composite literals
	dup("func() {}"),
	dup("[]int{}"),
	{"func(x int) complex128 { return 0 }", "func(x int) complex128 {…}"},
	{"[]int{1, 2, 3}", "[]int{…}"},

	// type expressions
	dup("[1 << 10]byte"),
	dup("[]int"),
	dup("*int"),
	dup("struct{x int}"),
	dup("func()"),
	dup("func(int, float32) string"),
	dup("interface{m()}"),
	dup("interface{m() string; n(x int)}"),
	dup("interface{~int}"),
	dup("interface{~int | ~float64 | ~string}"),
	dup("interface{~int; m()}"),
	dup("interface{~int | ~float64 | ~string; m() string; n(x int)}"),
	dup("map[string]int"),
	dup("chan E"),
	dup("<-chan E"),
	dup("chan<- E"),

	// new interfaces
	dup("interface{int}"),
	dup("interface{~int}"),

	// generic constraints
	dup("interface{~a | ~b | ~c; ~int | ~string; float64; m()}"),
	dup("interface{int | string}"),
	dup("interface{~int | ~string; float64; m()}"),
	dup("interface{~T[int, string] | string}"),

	// generic types
	dup("x[T]"),
	dup("x[N | A | S]"),
	dup("x[N, A]"),

	// non-type expressions
	dup("(x)"),
	dup("x.f"),
	dup("a[i]"),

	dup("s[:]"),
	dup("s[i:]"),
	dup("s[:j]"),
	dup("s[i:j]"),
	dup("s[:j:k]"),
	dup("s[i:j:k]"),

	dup("x.(T)"),

	dup("x.([10]int)"),
	dup("x.([...]int)"),

	dup("x.(struct{})"),
	dup("x.(struct{x int; y, z float32; E})"),

	dup("x.(func())"),
	dup("x.(func(x int))"),
	dup("x.(func() int)"),
	dup("x.(func(x, y int, z float32) (r int))"),
	dup("x.(func(a, b, c int))"),
	dup("x.(func(x ...T))"),

	dup("x.(interface{})"),
	dup("x.(interface{m(); n(x int); E})"),
	dup("x.(interface{m(); n(x int) T; E; F})"),

	dup("x.(map[K]V)"),

	dup("x.(chan E)"),
	dup("x.(<-chan E)"),
	dup("x.(chan<- chan int)"),
	dup("x.(chan<- <-chan int)"),
	dup("x.(<-chan chan int)"),
	dup("x.(chan (<-chan int))"),

	dup("f()"),
	dup("f(x)"),
	dup("int(x)"),
	dup("f(x, x + y)"),
	dup("f(s...)"),
	dup("f(a, s...)"),

	// generic functions
	dup("f[T]()"),
	dup("f[T](T)"),
	dup("f[T, T1]()"),
	dup("f[T, T1](T, T1)"),

	dup("*x"),
	dup("&x"),
	dup("x + y"),
	dup("x + y << (2 * s)"),
}

func TestShortString(t *testing.T) {
	for _, test := range exprTests {
		src := "package p; var _ = " + test[0]
		ast, err := Parse(nil, strings.NewReader(src), nil, nil, 0)
		if err != nil {
			t.Errorf("%s: %s", test[0], err)
			continue
		}
		x := ast.DeclList[0].(*VarDecl).Values
		if got := String(x); got != test[1] {
			t.Errorf("%s: got %s, want %s", test[0], got, test[1])
		}
	}
}
