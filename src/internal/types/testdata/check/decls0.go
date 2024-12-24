// -lang=go1.17

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// type declarations

package p // don't permit non-interface elements in interfaces

import "unsafe"

const pi = 3.1415

type (
	N undefined /* ERROR "undefined" */
	B bool
	I int32
	A [10]P
	T struct {
		x, y P
	}
	P *T
	R (*R)
	F func(A) I
	Y interface {
		f(A) I
	}
	S [](((P)))
	M map[I]F
	C chan<- I

	// blank types must be typechecked
	_ pi /* ERROR "not a type" */
	_ struct{}
	_ struct{ pi /* ERROR "not a type" */ }
)


// declarations of init
const _, init /* ERROR "cannot declare init" */ , _ = 0, 1, 2
type init /* ERROR "cannot declare init" */ struct{}
var _, init /* ERROR "cannot declare init" */ int

func init() {}
func init /* ERROR "missing function body" */ ()

func _() { const init = 0 }
func _() { type init int }
func _() { var init int; _ = init }

// invalid array types
type (
	iA0 [... /* ERROR "invalid use of [...] array" */ ]byte
	// The error message below could be better. At the moment
	// we believe an integer that is too large is not an integer.
	// But at least we get an error.
	iA1 [1 /* ERROR "invalid array length" */ <<100]int
	iA2 [- /* ERROR "invalid array length" */ 1]complex128
	iA3 ["foo" /* ERROR "must be integer" */ ]string
	iA4 [float64 /* ERROR "must be integer" */ (0)]int
)


type (
	p1 pi.foo /* ERROR "pi.foo is not a type" */
	p2 unsafe.Pointer
)


type (
	Pi pi /* ERROR "not a type" */

	a /* ERROR "invalid recursive type" */ a
	a /* ERROR "redeclared" */ int

	b /* ERROR "invalid recursive type" */ c
	c d
	d e
	e b

	t *t

	U V
	V *W
	W U

	P1 *S2
	P2 P1

	S0 struct {
	}
	S1 struct {
		a, b, c int
		u, v, a /* ERROR "redeclared" */ float32
	}
	S2 struct {
		S0 // embedded field
		S0 /* ERROR "redeclared" */ int
	}
	S3 struct {
		x S2
	}
	S4/* ERROR "invalid recursive type" */ struct {
		S4
	}
	S5 /* ERROR "invalid recursive type" */ struct {
		S6
	}
	S6 struct {
		field S7
	}
	S7 struct {
		S5
	}

	L1 []L1
	L2 []int

	A1 [10.0]int
	A2 /* ERROR "invalid recursive type" */ [10]A2
	A3 /* ERROR "invalid recursive type" */ [10]struct {
		x A4
	}
	A4 [10]A3

	F1 func()
	F2 func(x, y, z float32)
	F3 func(x, y, x /* ERROR "redeclared" */ float32)
	F4 func() (x, y, x /* ERROR "redeclared" */ float32)
	F5 func(x int) (x /* ERROR "redeclared" */ float32)
	F6 func(x ...int)

	I1 interface{}
	I2 interface {
		m1()
	}
	I3 interface {
		m1()
		m1 /* ERROR "duplicate method m1" */ ()
	}
	I4 interface {
		m1(x, y, x /* ERROR "redeclared" */ float32)
		m2() (x, y, x /* ERROR "redeclared" */ float32)
		m3(x int) (x /* ERROR "redeclared" */ float32)
	}
	I5 interface {
		m1(I5)
	}
	I6 interface {
		S0 /* ERROR "non-interface type S0" */
	}
	I7 interface {
		I1
		I1
	}
	I8 /* ERROR "invalid recursive type" */ interface {
		I8
	}
	I9 /* ERROR "invalid recursive type" */ interface {
		I10
	}
	I10 interface {
		I11
	}
	I11 interface {
		I9
	}

	C1 chan int
	C2 <-chan int
	C3 chan<- C3
	C4 chan C5
	C5 chan C6
	C6 chan C4

	M1 map[Last]string
	M2 map[string]M2

	Last int
)

// cycles in function/method declarations
// (test cases for issues #5217, #25790 and variants)
func f1(x f1 /* ERROR "not a type" */ ) {}
func f2(x *f2 /* ERROR "not a type" */ ) {}
func f3() (x f3 /* ERROR "not a type" */ ) { return }
func f4() (x *f4 /* ERROR "not a type" */ ) { return }
// TODO(#43215) this should be detected as a cycle error
func f5([unsafe.Sizeof(f5)]int) {}

func (S0) m1 (x S0.m1 /* ERROR "S0.m1 is not a type" */ ) {}
func (S0) m2 (x *S0.m2 /* ERROR "S0.m2 is not a type" */ ) {}
func (S0) m3 () (x S0.m3 /* ERROR "S0.m3 is not a type" */ ) { return }
func (S0) m4 () (x *S0.m4 /* ERROR "S0.m4 is not a type" */ ) { return }

// interfaces may not have any blank methods
type BlankI interface {
	_ /* ERROR "methods must have a unique non-blank name" */ ()
	_ /* ERROR "methods must have a unique non-blank name" */ (float32) int
	m()
}

// non-interface types may have multiple blank methods
type BlankT struct{}

func (BlankT) _() {}
func (BlankT) _(int) {}
func (BlankT) _() int { return 0 }
func (BlankT) _(int) int { return 0}
