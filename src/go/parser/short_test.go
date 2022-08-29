// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains test cases for short valid and invalid programs.

package parser

import "testing"

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
	`package p; type _ struct{ int }`,
	`package p; type _ struct{ pkg.T }`,
	`package p; type _ struct{ *pkg.T }`,
	`package p; var _ = func()T(nil)`,
	`package p; func _(T (P))`,
	`package p; func _(T []E)`,
	`package p; func _(T [P]E)`,
	`package p; type _ [A+B]struct{}`,
	`package p; func (R) _()`,
	`package p; type _ struct{ f [n]E }`,
	`package p; type _ struct{ f [a+b+c+d]E }`,
	`package p; type I1 interface{}; type I2 interface{ I1 }`,

	// generic code
	`package p; type _ []T[int]`,
	`package p; type T[P any] struct { P }`,
	`package p; type T[P comparable] struct { P }`,
	`package p; type T[P comparable[P]] struct { P }`,
	`package p; type T[P1, P2 any] struct { P1; f []P2 }`,
	`package p; func _[T any]()()`,
	`package p; func _(T (P))`,
	`package p; func f[A, B any](); func _() { _ = f[int, int] }`,
	`package p; func _(x T[P1, P2, P3])`,
	`package p; func _(x p.T[Q])`,
	`package p; func _(p.T[Q])`,
	`package p; type _[A interface{},] struct{}`,
	`package p; type _[A interface{}] struct{}`,
	`package p; type _[A,  B any,] struct{}`,
	`package p; type _[A, B any] struct{}`,
	`package p; type _[A any,] struct{}`,
	`package p; type _[A any]struct{}`,
	`package p; type _[A any] struct{ A }`,
	`package p; func _[T any]()`,
	`package p; func _[T any](x T)`,
	`package p; func _[T1, T2 any](x T)`,
	`package p; func _[A, B any](a A) B`,
	`package p; func _[A, B C](a A) B`,
	`package p; func _[A, B C[A, B]](a A) B`,

	`package p; type _[A, B any] interface { _(a A) B }`,
	`package p; type _[A, B C[A, B]] interface { _(a A) B }`,
	`package p; func _[T1, T2 interface{}](x T1) T2`,
	`package p; func _[T1 interface{ m() }, T2, T3 interface{}](x T1, y T3) T2`,
	`package p; var _ = []T[int]{}`,
	`package p; var _ = [10]T[int]{}`,
	`package p; var _ = func()T[int]{}`,
	`package p; var _ = map[T[int]]T[int]{}`,
	`package p; var _ = chan T[int](x)`,
	`package p; func _(_ T[P], T P) T[P]`,
	`package p; var _ T[chan int]`,

	`package p; func (_ R[P]) _(x T)`,
	`package p; func (_ R[ P, Q]) _(x T)`,

	`package p; func (R[P]) _()`,
	`package p; func _(T[P])`,
	`package p; func _(T[P1, P2, P3 ])`,
	`package p; func _(T[P]) T[P]`,
	`package p; type _ struct{ T[P]}`,
	`package p; type _ struct{ T[struct{a, b, c int}] }`,
	`package p; type _ interface{int|float32; bool; m(); string;}`,
	`package p; type I1[T any] interface{}; type I2 interface{ I1[int] }`,
	`package p; type I1[T any] interface{}; type I2[T any] interface{ I1[T] }`,
	`package p; type _ interface { N[T] }`,
	`package p; type T[P any] = T0`,
}

func TestValid(t *testing.T) {
	for _, src := range valids {
		checkErrors(t, src, src, DeclarationErrors|AllErrors, false)
	}
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
	`package p; func f() { _ = (<-<- /* ERROR "expected 'chan'" */ chan int)(nil) };`,
	`package p; func f() { _ = (<-chan<-chan<-chan<-chan<-chan<- /* ERROR "expected channel type" */ int)(nil) };`,
	`package p; func f() { if x := g(); x /* ERROR "expected boolean expression" */ = 0 {}};`,
	`package p; func f() { _ = x = /* ERROR "expected '=='" */ 0 {}};`,
	`package p; func f() { _ = 1 == func()int { var x bool; x = x = /* ERROR "expected '=='" */ true; return x }() };`,
	`package p; func f() { var s []int; _ = s[] /* ERROR "expected operand" */ };`,
	`package p; func f() { var s []int; _ = s[i:j: /* ERROR "final index required" */ ] };`,
	`package p; func f() { var s []int; _ = s[i: /* ERROR "middle index required" */ :k] };`,
	`package p; func f() { var s []int; _ = s[i: /* ERROR "middle index required" */ :] };`,
	`package p; func f() { var s []int; _ = s[: /* ERROR "middle index required" */ :] };`,
	`package p; func f() { var s []int; _ = s[: /* ERROR "middle index required" */ ::] };`,
	`package p; func f() { var s []int; _ = s[i:j:k: /* ERROR "expected ']'" */ l] };`,
	`package p; func f() { for x /* ERROR "boolean or range expression" */ = []string {} }`,
	`package p; func f() { for x /* ERROR "boolean or range expression" */ := []string {} }`,
	`package p; func f() { for i /* ERROR "boolean or range expression" */ , x = []string {} }`,
	`package p; func f() { for i /* ERROR "boolean or range expression" */ , x := []string {} }`,
	`package p; func f() { go f /* ERROR HERE "must be function call" */ }`,
	`package p; func f() { go ( /* ERROR "must not be parenthesized" */ f()) }`,
	`package p; func f() { defer func() {} /* ERROR HERE "must be function call" */ }`,
	`package p; func f() { defer ( /* ERROR "must not be parenthesized" */ f()) }`,
	`package p; func f() { go func() { func() { f(x func /* ERROR "missing ','" */ (){}) } } }`,
	`package p; func _() (type /* ERROR "found 'type'" */ T)(T)`,
	`package p; func (type /* ERROR "found 'type'" */ T)(T) _()`,
	`package p; type _[A+B, /* ERROR "unexpected comma" */ ] int`,

	`package p; type _ struct{ [ /* ERROR "expected '}', found '\['" */ ]byte }`,
	`package p; type _ struct{ ( /* ERROR "cannot parenthesize embedded type" */ int) }`,
	`package p; type _ struct{ ( /* ERROR "cannot parenthesize embedded type" */ []byte) }`,
	`package p; type _ struct{ *( /* ERROR "cannot parenthesize embedded type" */ int) }`,
	`package p; type _ struct{ *( /* ERROR "cannot parenthesize embedded type" */ []byte) }`,

	// issue 8656
	`package p; func f() (a b string /* ERROR "missing ','" */ , ok bool)`,

	// issue 9639
	`package p; var x, y, z; /* ERROR "expected type" */`,

	// issue 12437
	`package p; var _ = struct { x int, /* ERROR "expected ';', found ','" */ }{};`,
	`package p; var _ = struct { x int, /* ERROR "expected ';', found ','" */ y float }{};`,

	// issue 11611
	`package p; type _ struct { int, } /* ERROR "expected 'IDENT', found '}'" */ ;`,
	`package p; type _ struct { int, float } /* ERROR "expected type, found '}'" */ ;`,

	// issue 13475
	`package p; func f() { if true {} else ; /* ERROR "expected if statement or block" */ }`,
	`package p; func f() { if true {} else defer /* ERROR "expected if statement or block" */ f() }`,

	// generic code
	`package p; type _[_ any] int; var _ = T[] /* ERROR "expected operand" */ {}`,
	`package p; var _ func[ /* ERROR "must have no type parameters" */ T any](T)`,
	`package p; func _[]/* ERROR "empty type parameter list" */()`,

	// TODO(rfindley) a better location would be after the ']'
	`package p; type _[A /* ERROR "all type parameters must be named" */ ,] struct{ A }`,

	// TODO(rfindley) this error is confusing.
	`package p; func _[type /* ERROR "all type parameters must be named" */ P, *Q interface{}]()`,

	`package p; func (T) _[ /* ERROR "must have no type parameters" */ A, B any](a A) B`,
	`package p; func (T) _[ /* ERROR "must have no type parameters" */ A, B C](a A) B`,
	`package p; func (T) _[ /* ERROR "must have no type parameters" */ A, B C[A, B]](a A) B`,

	`package p; func(*T[e, e /* ERROR "e redeclared" */ ]) _()`,
}

func TestInvalid(t *testing.T) {
	for _, src := range invalids {
		checkErrors(t, src, src, DeclarationErrors|AllErrors, true)
	}
}
