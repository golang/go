// -lang=go1.17

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p // don't permit non-interface elements in interfaces

import (
	"fmt"
	syn "regexp/syntax"
	t1 "text/template"
	t2 "html/template"
)

func issue7035() {
	type T struct{ X int }
	_ = func() {
		fmt.Println() // must refer to imported fmt rather than the fmt below
	}
	fmt := new(T)
	_ = fmt.X
}

func issue8066() {
	const (
		_ = float32(340282356779733661637539395458142568447)
		_ = float32(340282356779733661637539395458142568448 /* ERROR "cannot convert" */ )
	)
}

// Check that a missing identifier doesn't lead to a spurious error cascade.
func issue8799a() {
	x, ok := missing /* ERROR "undefined" */ ()
	_ = !ok
	_ = x
}

func issue8799b(x int, ok bool) {
	x, ok = missing /* ERROR "undefined" */ ()
	_ = !ok
	_ = x
}

func issue9182() {
	type Point C /* ERROR "undefined" */ .Point
	// no error for composite literal based on unknown type
	_ = Point{x: 1, y: 2}
}

func f0() (a []int)         { return }
func f1() (a []int, b int)  { return }
func f2() (a, b []int)      { return }

func append_([]int, ...int) {}

func issue9473(a []int, b ...int) {
	// variadic builtin function
	_ = append(f0())
	_ = append(f0(), f0()...)
	_ = append(f1())
	_ = append(f2 /* ERRORx `cannot use .* in argument` */ ())
	_ = append(f2()... /* ERROR "cannot use ..." */ )
	_ = append(f0(), f1 /* ERROR "multiple-value f1" */ ())
	_ = append(f0(), f2 /* ERROR "multiple-value f2" */ ())
	_ = append(f0(), f1 /* ERROR "multiple-value f1" */ ()...)
	_ = append(f0(), f2 /* ERROR "multiple-value f2" */ ()...)

	// variadic user-defined function
	append_(f0())
	append_(f0(), f0()...)
	append_(f1())
	append_(f2 /* ERRORx `cannot use .* in argument` */ ())
	append_(f2()... /* ERROR "cannot use ..." */ )
	append_(f0(), f1 /* ERROR "multiple-value f1" */ ())
	append_(f0(), f2 /* ERROR "multiple-value f2" */ ())
	append_(f0(), f1 /* ERROR "multiple-value f1" */ ()...)
	append_(f0(), f2 /* ERROR "multiple-value f2" */ ()...)
}

// Check that embedding a non-interface type in an interface results in a good error message.
func issue10979() {
	type _ interface {
		int /* ERROR "non-interface type int" */
	}
	type T struct{}
	type _ interface {
		T /* ERROR "non-interface type T" */
	}
	type _ interface {
		nosuchtype /* ERROR "undefined: nosuchtype" */
	}
	type _ interface {
		fmt.Nosuchtype /* ERROR "undefined: fmt.Nosuchtype" */
	}
	type _ interface {
		nosuchpkg /* ERROR "undefined: nosuchpkg" */ .Nosuchtype
	}
	type I interface {
		I.m /* ERROR "I.m is not a type" */
		m()
	}
}

// issue11347
// These should not crash.
var a1, b1, c1 /* ERROR "cycle" */ b1 /* ERROR "b1 is not a type" */ = 0 > 0<<""[""[c1]]>c1
var a2, b2 /* ERROR "cycle" */ = 0 /* ERROR "assignment mismatch" */ /* ERROR "assignment mismatch" */ > 0<<""[b2]
var a3, b3 /* ERROR "cycle" */ = int /* ERROR "assignment mismatch" */ /* ERROR "assignment mismatch" */ (1<<""[b3])

// issue10260
// Check that error messages explain reason for interface assignment failures.
type (
	I0 interface{}
	I1 interface{ foo() }
	I2 interface{ foo(x int) }
	T0 struct{}
	T1 struct{}
	T2 struct{}
)

func (*T1) foo() {}
func (*T2) foo(x int) {}

func issue10260() {
	var (
		i0 I0
		i1 I1
		i2 I2
		t0 *T0
		t1 *T1
		t2 *T2
	)

	var x I1
	x = T1 /* ERRORx `cannot use T1{} .* as I1 value in assignment: T1 does not implement I1 \(method foo has pointer receiver\)` */ {}
	_ = x /* ERROR "impossible type assertion: x.(T1)\n\tT1 does not implement I1 (method foo has pointer receiver)" */ .(T1)

	T1{}.foo /* ERROR "cannot call pointer method foo on T1" */ ()
	x.Foo /* ERROR "x.Foo undefined (type I1 has no field or method Foo, but does have method foo)" */ ()

	_ = i2 /* ERROR "impossible type assertion: i2.(*T1)\n\t*T1 does not implement I2 (wrong type for method foo)\n\t\thave foo()\n\t\twant foo(int)" */ .(*T1)

	i1 = i0 /* ERRORx `cannot use i0 .* as I1 value in assignment: I0 does not implement I1 \(missing method foo\)` */
	i1 = t0 /* ERRORx `.* t0 .* as I1 .*: \*T0 does not implement I1 \(missing method foo\)` */
	i1 = i2 /* ERRORx `.* i2 .* as I1 .*: I2 does not implement I1 \(wrong type for method foo\)\n\t\thave foo\(int\)\n\t\twant foo\(\)` */
	i1 = t2 /* ERRORx `.* t2 .* as I1 .*: \*T2 does not implement I1 \(wrong type for method foo\)\n\t\thave foo\(int\)\n\t\twant foo\(\)` */
	i2 = i1 /* ERRORx `.* i1 .* as I2 .*: I1 does not implement I2 \(wrong type for method foo\)\n\t\thave foo\(\)\n\t\twant foo\(int\)` */
	i2 = t1 /* ERRORx `.* t1 .* as I2 .*: \*T1 does not implement I2 \(wrong type for method foo\)\n\t\thave foo\(\)\n\t\twant foo\(int\)` */

	_ = func() I1 { return i0 /* ERRORx `cannot use i0 .* as I1 value in return statement: I0 does not implement I1 \(missing method foo\)` */ }
	_ = func() I1 { return t0 /* ERRORx `.* t0 .* as I1 .*: \*T0 does not implement I1 \(missing method foo\)` */ }
	_ = func() I1 { return i2 /* ERRORx `.* i2 .* as I1 .*: I2 does not implement I1 \(wrong type for method foo\)\n\t\thave foo\(int\)\n\t\twant foo\(\)` */ }
	_ = func() I1 { return t2 /* ERRORx `.* t2 .* as I1 .*: \*T2 does not implement I1 \(wrong type for method foo\)\n\t\thave foo\(int\)\n\t\twant foo\(\)` */ }
	_ = func() I2 { return i1 /* ERRORx `.* i1 .* as I2 .*: I1 does not implement I2 \(wrong type for method foo\)\n\t\thave foo\(\)\n\t\twant foo\(int\)` */ }
	_ = func() I2 { return t1 /* ERRORx `.* t1 .* as I2 .*: \*T1 does not implement I2 \(wrong type for method foo\)\n\t\thave foo\(\)\n\t\twant foo\(int\)` */ }

	// a few more - less exhaustive now

	f := func(I1, I2){}
	f(i0 /* ERROR "missing method foo" */ , i1 /* ERROR "wrong type for method foo" */ )

	_ = [...]I1{i0 /* ERRORx `cannot use i0 .* as I1 value in array or slice literal: I0 does not implement I1 \(missing method foo\)` */ }
	_ = [...]I1{i2 /* ERRORx `cannot use i2 .* as I1 value in array or slice literal: I2 does not implement I1 \(wrong type for method foo\)\n\t\thave foo\(int\)\n\t\twant foo\(\)` */ }
	_ = []I1{i0 /* ERROR "missing method foo" */ }
	_ = []I1{i2 /* ERROR "wrong type for method foo" */ }
	_ = map[int]I1{0: i0 /* ERROR "missing method foo" */ }
	_ = map[int]I1{0: i2 /* ERROR "wrong type for method foo" */ }

	make(chan I1) <- i0 /* ERROR "missing method foo" */
	make(chan I1) <- i2 /* ERROR "wrong type for method foo" */
}

// Check that constants representable as integers are in integer form
// before being used in operations that are only defined on integers.
func issue14229() {
	// from the issue
	const _ = int64(-1<<63) % 1e6

	// related
	const (
		a int = 3
		b = 4.0
		_ = a / b
		_ = a % b
		_ = b / a
		_ = b % a
	)
}

// Check that in a n:1 variable declaration with type and initialization
// expression the type is distributed to all variables of the lhs before
// the initialization expression assignment is checked.
func issue15755() {
	// from issue
	var i interface{}
	type b bool
	var x, y b = i.(b)
	_ = x == y

	// related: we should see an error since the result of f1 is ([]int, int)
	var u, v []int = f1 /* ERROR "cannot use f1" */ ()
	_ = u
	_ = v
}

// Test that we don't get "declared and not used"
// errors in the context of invalid/C objects.
func issue20358() {
	var F C /* ERROR "undefined" */ .F
	var A C /* ERROR "undefined" */ .A
	var S C /* ERROR "undefined" */ .S
	type T C /* ERROR "undefined" */ .T
	type P C /* ERROR "undefined" */ .P

	// these variables must be "used" even though
	// the LHS expressions/types below in which
	// context they are used are unknown/invalid
	var f, a, s1, s2, s3, t, p int

	_ = F(f)
	_ = A[a]
	_ = S[s1:s2:s3]
	_ = T{t}
	_ = P{f: p}
}

// Test that we don't declare lhs variables in short variable
// declarations before we type-check function literals on the
// rhs.
func issue24026() {
	f := func() int { f(0) /* must refer to outer f */; return 0 }
	_ = f

	_ = func() {
		f := func() { _ = f() /* must refer to outer f */ }
		_ = f
	}

	// b and c must not be visible inside function literal
	a := 0
	a, b, c := func() (int, int, int) {
		return a, b /* ERROR "undefined" */ , c /* ERROR "undefined" */
	}()
	_, _ = b, c
}

func f(int) {} // for issue24026

// Test that we don't report a "missing return statement" error
// (due to incorrect context when type-checking interfaces).
func issue24140(x interface{}) int {
        switch x.(type) {
        case interface{}:
                return 0
        default:
                panic(0)
        }
}

// Test that we don't crash when the 'if' condition is missing.
func issue25438() {
	if { /* ERROR "missing condition" */ }
	if x := 0; /* ERROR "missing condition" */ { _ = x }
	if
	{ /* ERROR "missing condition" */ }
}

// Test that we can embed alias type names in interfaces.
type issue25301 interface {
	E
}

type E = interface {
	m()
}

// Test case from issue.
// cmd/compile reports a cycle as well.
type issue25301b /* ERROR "invalid recursive type" */ = interface {
	m() interface{ issue25301b }
}

type issue25301c interface {
	notE // ERRORx "non-interface type (struct{}|notE)"
}

type notE = struct{}

// Test that method declarations don't introduce artificial cycles
// (issue #26124).
const CC TT = 1
type TT int
func (TT) MM() [CC]TT

// Reduced test case from issue #26124.
const preloadLimit LNumber = 128
type LNumber float64
func (LNumber) assertFunction() *LFunction
type LFunction struct {
	GFunction LGFunction
}
type LGFunction func(*LState)
type LState struct {
	reg *registry
}
type registry struct {
	alloc *allocator
}
type allocator struct {
	_ [int(preloadLimit)]int
}

// Test that we don't crash when type-checking composite literals
// containing errors in the type.
var issue27346 = [][n /* ERROR "undefined" */ ]int{
	0: {},
}

var issue22467 = map[int][... /* ERROR "invalid use of [...] array" */ ]int{0: {}}

// Test that invalid use of ... in parameter lists is recognized
// (issue #28281).
func issue28281a(int, int, ...int)
func issue28281b(a, b int, c ...int)
func issue28281c(a, b, c ... /* ERROR "can only use ... with final parameter" */ int)
func issue28281d(... /* ERROR "can only use ... with final parameter" */ int, int)
func issue28281e(a, b, c  ... /* ERROR "can only use ... with final parameter" */ int, d int)
func issue28281f(... /* ERROR "can only use ... with final parameter" */ int, ... /* ERROR "can only use ... with final parameter" */ int, int)
func (... /* ERROR "invalid use of '...'" */ TT) f()
func issue28281g() (... /* ERROR "can only use ... with final parameter" */ TT)

// Issue #26234: Make various field/method lookup errors easier to read by matching cmd/compile's output
func issue26234a(f *syn.Prog) {
	// The error message below should refer to the actual package name (syntax)
	// not the local package name (syn).
	f.foo /* ERROR "f.foo undefined (type *syntax.Prog has no field or method foo)" */
}

type T struct {
	x int
	E1
	E2
}

type E1 struct{ f int }
type E2 struct{ f int }

func issue26234b(x T) {
	_ = x.f /* ERROR "ambiguous selector x.f" */
}

func issue26234c() {
	T.x /* ERROR "T.x undefined (type T has no method x)" */ ()
}

func issue35895() {
	// T is defined in this package, don't qualify its name with the package name.
	var _ T = 0 // ERROR "cannot use 0 (untyped int constant) as T"

	// There is only one package with name syntax imported, only use the (global) package name in error messages.
	var _ *syn.Prog = 0 // ERROR "cannot use 0 (untyped int constant) as *syntax.Prog"

	// Because both t1 and t2 have the same global package name (template),
	// qualify packages with full path name in this case.
	var _ t1.Template = t2 /* ERRORx `cannot use .* \(value of type .html/template.\.Template\) as .text/template.\.Template` */ .Template{}
}

func issue42989(s uint) {
	var m map[int]string
	delete(m, 1<<s)
	delete(m, 1.<<s)
}
