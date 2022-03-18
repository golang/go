// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// type aliases

package decls4

type (
	T0 [10]int
	T1 []byte
	T2 struct {
		x int
	}
	T3 interface{
		m() T2
	}
	T4 func(int, T0) chan T2
)

type (
	Ai = int
	A0 = T0
	A1 = T1
	A2 = T2
	A3 = T3
	A4 = T4

	A10 = [10]int
	A11 = []byte
	A12 = struct {
		x int
	}
	A13 = interface{
		m() A2
	}
	A14 = func(int, A0) chan A2
)

// check assignment compatibility due to equality of types
var (
	xi_ int
	ai Ai = xi_

	x0 T0
	a0 A0 = x0

	x1 T1
	a1 A1 = x1

	x2 T2
	a2 A2 = x2

	x3 T3
	a3 A3 = x3

	x4 T4
	a4 A4 = x4
)

// alias receiver types
func (Ai /* ERROR "invalid receiver" */) m1() {}
func (T0) m1() {}
func (A0) m1 /* ERROR already declared */ () {}
func (A0) m2 () {}
func (A3 /* ERROR invalid receiver */ ) m1 () {}
func (A10 /* ERROR invalid receiver */ ) m1() {}

// x0 has methods m1, m2 declared via receiver type names T0 and A0
var _ interface{ m1(); m2() } = x0

// alias receiver types (test case for issue #23042)
type T struct{}

var (
	_ = T.m
	_ = T{}.m
	_ interface{m()} = T{}
)

var (
	_ = T.n
	_ = T{}.n
	_ interface{m(); n()} = T{}
)

type U = T
func (U) m() {}

// alias receiver types (long type declaration chains)
type (
	V0 = V1
	V1 = (V2)
	V2 = ((V3))
	V3 = T
)

func (V0) m /* ERROR already declared */ () {}
func (V1) n() {}

// alias receiver types (invalid due to cycles)
type (
	W0 /* ERROR illegal cycle */ = W1
	W1 = (W2)
	W2 = ((W0))
)

func (W0) m() {} // no error expected (due to above cycle error)
func (W1) n() {}

// alias receiver types (invalid due to builtin underlying type)
type (
	B0 = B1
	B1 = B2
	B2 = int
)

func (B0 /* ERROR invalid receiver */ ) m() {}
func (B1 /* ERROR invalid receiver */ ) n() {}

// cycles
type (
	C2 /* ERROR illegal cycle */ = C2
	C3 /* ERROR illegal cycle */ = C4
	C4 = C3
	C5 struct {
		f *C6
	}
	C6 = C5
	C7 /* ERROR illegal cycle */  struct {
		f C8
	}
	C8 = C7
)

// embedded fields
var (
	s0 struct { T0 }
	s1 struct { A0 } = s0 /* ERROR cannot use */ // embedded field names are different
)

// embedding and lookup of fields and methods
func _(s struct{A0}) { s.A0 = x0 }

type eX struct{xf int}

func (eX) xm()

type eY = struct{eX} // field/method set of eY includes xf, xm

type eZ = *struct{eX} // field/method set of eZ includes xf, xm

type eA struct {
	eX // eX contributes xf, xm to eA
}

type eA2 struct {
	*eX // *eX contributes xf, xm to eA
}

type eB struct {
	eY // eY contributes xf, xm to eB
}

type eB2 struct {
	*eY // *eY contributes xf, xm to eB
}

type eC struct {
	eZ // eZ contributes xf, xm to eC
}

var (
	_ = eA{}.xf
	_ = eA{}.xm
	_ = eA2{}.xf
	_ = eA2{}.xm
	_ = eB{}.xf
	_ = eB{}.xm
	_ = eB2{}.xf
	_ = eB2{}.xm
	_ = eC{}.xf
	_ = eC{}.xm
)

// ambiguous selectors due to embedding via type aliases
type eD struct {
	eY
	eZ
}

var (
	_ = eD{}.xf /* ERROR ambiguous selector eD\{\}.xf */
	_ = eD{}.xm /* ERROR ambiguous selector eD\{\}.xm */
)

var (
	_ interface{ xm() } = eD /* ERROR missing method xm */ {}
)