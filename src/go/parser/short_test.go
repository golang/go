// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains test cases for short valid and invalid programs.

package parser

import (
	"go/internal/typeparams"
	"testing"
)

var valids = []string{
	"package p\n",
	`package p;`,
	`package p; import "fmt"; func f() { fmt.Println("Hello, World!") };`,
	`package p; func f() { if f(T{}) {} };`,
	`package p; func f() { _ = <-chan int(nil) };`,
	`package p; func f() { _ = (<-chan int)(nil) };`,
	`package p; func f() { _ = (<-chan <-chan int)(nil) };`,
	`package p; func f() { _ = <-chan <-chan <-chan <-chan <-int(nil) };`,
	`package p; func f(func() func() func());`,
	`package p; func f(...T);`,
	`package p; func f(float, ...int);`,
	`package p; func f(x int, a ...int) { f(0, a...); f(1, a...,) };`,
	`package p; func f(int,) {};`,
	`package p; func f(...int,) {};`,
	`package p; func f(x ...int,) {};`,
	`package p; type T []int; var a []bool; func f() { if a[T{42}[0]] {} };`,
	`package p; type T []int; func g(int) bool { return true }; func f() { if g(T{42}[0]) {} };`,
	`package p; type T []int; func f() { for _ = range []int{T{42}[0]} {} };`,
	`package p; var a = T{{1, 2}, {3, 4}}`,
	`package p; func f() { select { case <- c: case c <- d: case c <- <- d: case <-c <- d: } };`,
	`package p; func f() { select { case x := (<-c): } };`,
	`package p; func f() { if ; true {} };`,
	`package p; func f() { switch ; {} };`,
	`package p; func f() { for _ = range "foo" + "bar" {} };`,
	`package p; func f() { var s []int; g(s[:], s[i:], s[:j], s[i:j], s[i:j:k], s[:j:k]) };`,
	`package p; var ( _ = (struct {*T}).m; _ = (interface {T}).m )`,
	`package p; func ((T),) m() {}`,
	`package p; func ((*T),) m() {}`,
	`package p; func (*(T),) m() {}`,
	`package p; func _(x []int) { for range x {} }`,
	`package p; func _() { if [T{}.n]int{} {} }`,
	`package p; func _() { map[int]int{}[0]++; map[int]int{}[0] += 1 }`,
	`package p; func _(x interface{f()}) { interface{f()}(x).f() }`,
	`package p; func _(x chan int) { chan int(x) <- 0 }`,
	`package p; const (x = 0; y; z)`, // issue 9639
	`package p; var _ = map[P]int{P{}:0, {}:1}`,
	`package p; var _ = map[*P]int{&P{}:0, {}:1}`,
	`package p; type T = int`,
	`package p; type (T = p.T; _ = struct{}; x = *T)`,
	`package p; type T (*int)`,
	`package p; type _ struct{ ((int)) }`,
	`package p; type _ struct{ (*(int)) }`,
	`package p; type _ struct{ ([]byte) }`, // disallowed by type-checker
	`package p; var _ = func()T(nil)`,
	`package p; func _(T (P))`,
	`package p; func _(T []E)`,
	`package p; func _(T [P]E)`,
	`package p; type _ [A+B]struct{}`,
	`package p; func (R) _()`,
	`package p; type _ struct{ f [n]E }`,
	`package p; type _ struct{ f [a+b+c+d]E }`,
	`package p; type I1 interface{}; type I2 interface{ I1 }`,
}

// validWithTParamsOnly holds source code examples that are valid if
// parseTypeParams is set, but invalid if not. When checking with the
// parseTypeParams set, errors are ignored.
var validWithTParamsOnly = []string{
	`package p; type _ []T[ /* ERROR "expected ';', found '\['" */ int]`,
	`package p; type T[P any /* ERROR "expected ']', found any" */ ] struct { P }`,
	`package p; type T[P comparable /* ERROR "expected ']', found comparable" */ ] struct { P }`,
	`package p; type T[P comparable /* ERROR "expected ']', found comparable" */ [P]] struct { P }`,
	`package p; type T[P1, /* ERROR "expected ']', found ','" */ P2 any] struct { P1; f []P2 }`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ T any]()()`,
	`package p; func _(T (P))`,
	`package p; func f[ /* ERROR "expected '\(', found '\['" */ A, B any](); func _() { _ = f[int, int] }`,
	`package p; func _(x /* ERROR "mixed named and unnamed parameters" */ T[P1, P2, P3])`,
	`package p; func _(x /* ERROR "mixed named and unnamed parameters" */ p.T[Q])`,
	`package p; func _(p.T[ /* ERROR "missing ',' in parameter list" */ Q])`,
	`package p; type _[A interface /* ERROR "expected ']', found 'interface'" */ {},] struct{}`,
	`package p; type _[A interface /* ERROR "expected ']', found 'interface'" */ {}] struct{}`,
	`package p; type _[A, /* ERROR "expected ']', found ','" */  B any,] struct{}`,
	`package p; type _[A, /* ERROR "expected ']', found ','" */ B any] struct{}`,
	`package p; type _[A any /* ERROR "expected ']', found any" */,] struct{}`,
	`package p; type _[A any /* ERROR "expected ']', found any" */ ]struct{}`,
	`package p; type _[A any /* ERROR "expected ']', found any" */ ] struct{ A }`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ T any]()`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ T any](x T)`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ T1, T2 any](x T)`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ A, B any](a A) B`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ A, B C](a A) B`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ A, B C[A, B]](a A) B`,
	`package p; func (T) _[ /* ERROR "expected '\(', found '\['" */ A, B any](a A) B`,
	`package p; func (T) _[ /* ERROR "expected '\(', found '\['" */ A, B C](a A) B`,
	`package p; func (T) _[ /* ERROR "expected '\(', found '\['" */ A, B C[A, B]](a A) B`,
	`package p; type _[A, /* ERROR "expected ']', found ','" */ B any] interface { _(a A) B }`,
	`package p; type _[A, /* ERROR "expected ']', found ','" */ B C[A, B]] interface { _(a A) B }`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ T1, T2 interface{}](x T1) T2`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ T1 interface{ m() }, T2, T3 interface{}](x T1, y T3) T2`,
	`package p; var _ = [ /* ERROR "expected expression" */ ]T[int]{}`,
	`package p; var _ = [ /* ERROR "expected expression" */ 10]T[int]{}`,
	`package p; var _ = func /* ERROR "expected expression" */ ()T[int]{}`,
	`package p; var _ = map /* ERROR "expected expression" */ [T[int]]T[int]{}`,
	`package p; var _ = chan /* ERROR "expected expression" */ T[int](x)`,
	`package p; func _(_ T[ /* ERROR "missing ',' in parameter list" */ P], T P) T[P]`,
	`package p; var _ T[ /* ERROR "expected ';', found '\['" */ chan int]`,

	// TODO(rfindley) this error message could be improved.
	`package p; func (_ /* ERROR "mixed named and unnamed parameters" */ R[P]) _[T any](x T)`,
	`package p; func (_ /* ERROR "mixed named and unnamed parameters" */ R[ P, Q]) _[T1, T2 any](x T)`,

	`package p; func (R[P] /* ERROR "missing element type" */ ) _[T any]()`,
	`package p; func _(T[P] /* ERROR "missing element type" */ )`,
	`package p; func _(T[P1, /* ERROR "expected ']', found ','" */ P2, P3 ])`,
	`package p; func _(T[P] /* ERROR "missing element type" */ ) T[P]`,
	`package p; type _ struct{ T[P] /* ERROR "missing element type" */ }`,
	`package p; type _ struct{ T[struct /* ERROR "expected expression" */ {a, b, c int}] }`,
	`package p; type _ interface{type /* ERROR "expected '}', found 'type'" */ int}`,
	`package p; type _ interface{type /* ERROR "expected '}', found 'type'" */ int, float32; type bool; m(); type string;}`,
	`package p; type I1[T any /* ERROR "expected ']', found any" */ ] interface{}; type I2 interface{ I1[int] }`,
	`package p; type I1[T any /* ERROR "expected ']', found any" */ ] interface{}; type I2[T any] interface{ I1[T] }`,
	`package p; type _ interface { f[ /* ERROR "expected ';', found '\['" */ T any]() }`,
}

func TestValid(t *testing.T) {
	t.Run("no tparams", func(t *testing.T) {
		for _, src := range valids {
			checkErrors(t, src, src, DeclarationErrors|AllErrors, false)
		}
	})
	t.Run("tparams", func(t *testing.T) {
		if !typeparams.Enabled {
			t.Skip("type params are not enabled")
		}
		for _, src := range valids {
			checkErrors(t, src, src, DeclarationErrors|AllErrors, false)
		}
		for _, src := range validWithTParamsOnly {
			checkErrors(t, src, src, DeclarationErrors|AllErrors, false)
		}
	})
}

// TestSingle is useful to track down a problem with a single short test program.
func TestSingle(t *testing.T) {
	const src = `package p; var _ = T{}`
	checkErrors(t, src, src, DeclarationErrors|AllErrors, true)
}

var invalids = []string{
	`foo /* ERROR "expected 'package'" */ !`,
	`package p; func f() { if { /* ERROR "missing condition" */ } };`,
	`package p; func f() { if ; /* ERROR "missing condition" */ {} };`,
	`package p; func f() { if f(); /* ERROR "missing condition" */ {} };`,
	`package p; func f() { if _ = range /* ERROR "expected operand" */ x; true {} };`,
	`package p; func f() { switch _ /* ERROR "expected switch expression" */ = range x; true {} };`,
	`package p; func f() { for _ = range x ; /* ERROR "expected '{'" */ ; {} };`,
	`package p; func f() { for ; ; _ = range /* ERROR "expected operand" */ x {} };`,
	`package p; func f() { for ; _ /* ERROR "expected boolean or range expression" */ = range x ; {} };`,
	`package p; func f() { switch t = /* ERROR "expected ':=', found '='" */ t.(type) {} };`,
	`package p; func f() { switch t /* ERROR "expected switch expression" */ , t = t.(type) {} };`,
	`package p; func f() { switch t /* ERROR "expected switch expression" */ = t.(type), t {} };`,
	`package p; var a = [ /* ERROR "expected expression" */ 1]int;`,
	`package p; var a = [ /* ERROR "expected expression" */ ...]int;`,
	`package p; var a = struct /* ERROR "expected expression" */ {}`,
	`package p; var a = func /* ERROR "expected expression" */ ();`,
	`package p; var a = interface /* ERROR "expected expression" */ {}`,
	`package p; var a = [ /* ERROR "expected expression" */ ]int`,
	`package p; var a = map /* ERROR "expected expression" */ [int]int`,
	`package p; var a = chan /* ERROR "expected expression" */ int;`,
	`package p; var a = []int{[ /* ERROR "expected expression" */ ]int};`,
	`package p; var a = ( /* ERROR "expected expression" */ []int);`,
	`package p; var a = <- /* ERROR "expected expression" */ chan int;`,
	`package p; func f() { select { case _ <- chan /* ERROR "expected expression" */ int: } };`,
	`package p; func f() { _ = (<-<- /* ERROR "expected 'chan'" */ chan int)(nil) };`,
	`package p; func f() { _ = (<-chan<-chan<-chan<-chan<-chan<- /* ERROR "expected channel type" */ int)(nil) };`,
	`package p; func f() { var t []int; t /* ERROR "expected identifier on left side of :=" */ [0] := 0 };`,
	`package p; func f() { if x := g(); x /* ERROR "expected boolean expression" */ = 0 {}};`,
	`package p; func f() { _ = x = /* ERROR "expected '=='" */ 0 {}};`,
	`package p; func f() { _ = 1 == func()int { var x bool; x = x = /* ERROR "expected '=='" */ true; return x }() };`,
	`package p; func f() { var s []int; _ = s[] /* ERROR "expected operand" */ };`,
	`package p; func f() { var s []int; _ = s[i:j: /* ERROR "3rd index required" */ ] };`,
	`package p; func f() { var s []int; _ = s[i: /* ERROR "2nd index required" */ :k] };`,
	`package p; func f() { var s []int; _ = s[i: /* ERROR "2nd index required" */ :] };`,
	`package p; func f() { var s []int; _ = s[: /* ERROR "2nd index required" */ :] };`,
	`package p; func f() { var s []int; _ = s[: /* ERROR "2nd index required" */ ::] };`,
	`package p; func f() { var s []int; _ = s[i:j:k: /* ERROR "expected ']'" */ l] };`,
	`package p; func f() { for x /* ERROR "boolean or range expression" */ = []string {} }`,
	`package p; func f() { for x /* ERROR "boolean or range expression" */ := []string {} }`,
	`package p; func f() { for i /* ERROR "boolean or range expression" */ , x = []string {} }`,
	`package p; func f() { for i /* ERROR "boolean or range expression" */ , x := []string {} }`,
	`package p; func f() { go f /* ERROR HERE "function must be invoked" */ }`,
	`package p; func f() { defer func() {} /* ERROR HERE "function must be invoked" */ }`,
	`package p; func f() { go func() { func() { f(x func /* ERROR "missing ','" */ (){}) } } }`,
	`package p; func _() (type /* ERROR "found 'type'" */ T)(T)`,
	`package p; func (type /* ERROR "found 'type'" */ T)(T) _()`,
	`package p; type _[A+B, /* ERROR "expected ']'" */ ] int`,

	// TODO: this error should be positioned on the ':'
	`package p; var a = a[[]int:[ /* ERROR "expected expression" */ ]int];`,
	// TODO: the compiler error is better here: "cannot parenthesize embedded type"
	`package p; type I1 interface{}; type I2 interface{ (/* ERROR "expected '}', found '\('" */ I1) }`,

	// issue 8656
	`package p; func f() (a b string /* ERROR "missing ','" */ , ok bool)`,

	// issue 9639
	`package p; var x /* ERROR "missing variable type or initialization" */ , y, z;`,
	`package p; const x /* ERROR "missing constant value" */ ;`,
	`package p; const x /* ERROR "missing constant value" */ int;`,
	`package p; const (x = 0; y; z /* ERROR "missing constant value" */ int);`,

	// issue 12437
	`package p; var _ = struct { x int, /* ERROR "expected ';', found ','" */ }{};`,
	`package p; var _ = struct { x int, /* ERROR "expected ';', found ','" */ y float }{};`,

	// issue 11611
	`package p; type _ struct { int, } /* ERROR "expected 'IDENT', found '}'" */ ;`,
	`package p; type _ struct { int, float } /* ERROR "expected type, found '}'" */ ;`,

	// issue 13475
	`package p; func f() { if true {} else ; /* ERROR "expected if statement or block" */ }`,
	`package p; func f() { if true {} else defer /* ERROR "expected if statement or block" */ f() }`,
}

// invalidNoTParamErrs holds invalid source code examples annotated with the
// error messages produced when ParseTypeParams is not set.
var invalidNoTParamErrs = []string{
	`package p; type _[_ any /* ERROR "expected ']', found any" */ ] int; var _ = T[]{}`,
	`package p; type T[P any /* ERROR "expected ']', found any" */ ] = T0`,
	`package p; var _ func[ /* ERROR "expected '\(', found '\['" */ T any](T)`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ ]()`,
	`package p; type _[A, /* ERROR "expected ']', found ','" */] struct{ A }`,
	`package p; func _[ /* ERROR "expected '\(', found '\['" */ type P, *Q interface{}]()`,
}

// invalidTParamErrs holds invalid source code examples annotated with the
// error messages produced when ParseTypeParams is set.
var invalidTParamErrs = []string{
	`package p; type _[_ any] int; var _ = T[] /* ERROR "expected operand" */ {}`,
	`package p; type T[P any] = /* ERROR "cannot be alias" */ T0`,
	`package p; var _ func[ /* ERROR "cannot have type parameters" */ T any](T)`,
	`package p; func _[]/* ERROR "empty type parameter list" */()`,

	// TODO(rfindley) a better location would be after the ']'
	`package p; type _[A/* ERROR "all type parameters must be named" */,] struct{ A }`,

	// TODO(rfindley) this error is confusing.
	`package p; func _[type /* ERROR "all type parameters must be named" */P, *Q interface{}]()`,
}

func TestInvalid(t *testing.T) {
	t.Run("no tparams", func(t *testing.T) {
		for _, src := range invalids {
			checkErrors(t, src, src, DeclarationErrors|AllErrors|typeparams.DisallowParsing, true)
		}
		for _, src := range validWithTParamsOnly {
			checkErrors(t, src, src, DeclarationErrors|AllErrors|typeparams.DisallowParsing, true)
		}
		for _, src := range invalidNoTParamErrs {
			checkErrors(t, src, src, DeclarationErrors|AllErrors|typeparams.DisallowParsing, true)
		}
	})
	t.Run("tparams", func(t *testing.T) {
		if !typeparams.Enabled {
			t.Skip("type params are not enabled")
		}
		for _, src := range invalids {
			checkErrors(t, src, src, DeclarationErrors|AllErrors, true)
		}
		for _, src := range invalidTParamErrs {
			checkErrors(t, src, src, DeclarationErrors|AllErrors, true)
		}
	})
}
