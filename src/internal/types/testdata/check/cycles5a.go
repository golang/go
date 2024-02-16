// -gotypesalias=1

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

// test case from issue #18395

type (
	A interface { B }
	B interface { C }
	C interface { D; F() A }
	D interface { G() B }
)

var _ = A(nil).G // G must be found


// test case from issue #21804

type sourceBridge interface {
	listVersions() ([]Version, error)
}

type Constraint interface {
	copyTo(*ConstraintMsg)
}

type ConstraintMsg struct{}

func (m *ConstraintMsg) asUnpairedVersion() UnpairedVersion {
	return nil
}

type Version interface {
	Constraint
}

type UnpairedVersion interface {
	Version
}

var _ Constraint = UnpairedVersion(nil)


// derived test case from issue #21804

type (
	_ interface{ m(B1) }
	A1 interface{ a(D1) }
	B1 interface{ A1 }
	C1 interface{ B1 }
	D1 interface{ C1 }
)

var _ A1 = C1(nil)


// derived test case from issue #22701

func F(x I4) interface{} {
	return x.Method()
}

type Unused interface {
	RefersToI1(a I1)
}

type I1 interface {
	I2
	I3
}

type I2 interface {
	RefersToI4() I4
}

type I3 interface {
	Method() interface{}
}

type I4 interface {
	I1
}


// check embedding of error interface

type Error interface{ error }

var err Error
var _ = err.Error()


// more esoteric cases

type (
	T1 interface { T2 }
	T2 /* ERROR "invalid recursive type" */ T2
)

type (
	T3 interface { T4 }
	T4 /* ERROR "invalid recursive type" */ T5
	T5 = T6
	T6 = T7
	T7 = T4
)


// arbitrary code may appear inside an interface

const n = unsafe.Sizeof(func(){})

type I interface {
	m([unsafe.Sizeof(func() { I.m(nil, [n]byte{}) })]byte)
}


// test cases for varias alias cycles

type T10 /* ERROR "invalid recursive type" */ = *T10                 // issue #25141
type T11 /* ERROR "invalid recursive type" */ = interface{ f(T11) }  // issue #23139

// issue #18640
type (
	aa = bb
	bb struct {
		*aa
	}
)

type (
	a struct{ *b }
	b = c
	c struct{ *b }
)

// issue #24939
type (
	_ interface {
		M(P)
	}

	M interface {
		F() P
	}

	P = interface {
		I() M
	}
)

// issue #8699
type T12 /* ERROR "invalid recursive type" */ [len(a12)]int
var a12 = makeArray()
func makeArray() (res T12) { return }

// issue #20770
var r = newReader()
func newReader() r // ERROR "r is not a type"

// variations of the theme of #8699 and #20770
var arr /* ERROR "cycle" */ = f()
func f() [len(arr)]int

// issue #25790
func ff(ff /* ERROR "not a type" */ )
func gg((gg /* ERROR "not a type" */ ))

type T13 /* ERROR "invalid recursive type T13" */ [len(b13)]int
var b13 T13

func g1() [unsafe.Sizeof(g1)]int
func g2() [unsafe.Sizeof(x2)]int
var x2 = g2

// verify that we get the correct sizes for the functions above
// (note: assert is statically evaluated in go/types test mode)
func init() {
	assert(unsafe.Sizeof(g1) == 8)
	assert(unsafe.Sizeof(x2) == 8)
}

func h() [h /* ERROR "no value" */ ()[0]]int { panic(0) }

var c14 /* ERROR "cycle" */ T14
type T14 [uintptr(unsafe.Sizeof(&c14))]byte

// issue #34333
type T15 /* ERROR "invalid recursive type T15" */ struct {
	f func() T16
	b T16
}

type T16 struct {
	T15
}