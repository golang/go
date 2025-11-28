// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cycles

import "unsafe"

type (
	T0 int
	T1 /* ERROR "invalid recursive type: T1 refers to itself" */ T1
	T2 *T2

	T3 /* ERROR "invalid recursive type" */ T4
	T4 T5
	T5 T3

	T6 T7
	T7 *T8
	T8 T6

	// arrays
	A0 /* ERROR "invalid recursive type" */ [10]A0
	A1 [10]*A1

	A2 /* ERROR "invalid recursive type" */ [10]A3
	A3 [10]A4
	A4 A2

	A5 [10]A6
	A6 *A5

	// slices
	L0 []L0

	// structs
	S0 /* ERROR "invalid recursive type: S0 refers to itself" */ struct{ _ S0 }
	S1 /* ERROR "invalid recursive type: S1 refers to itself" */ struct{ S1 }
	S2 struct{ _ *S2 }
	S3 struct{ *S3 }

	S4 /* ERROR "invalid recursive type" */ struct{ S5 }
	S5 struct{ S6 }
	S6 S4

	// pointers
	P0 *P0
	PP /* ERROR "invalid recursive type" */ *struct{ PP.f }

	// functions
	F0 func(F0)
	F1 func() F1
	F2 func(F2) F2

	// interfaces
	I0 /* ERROR "invalid recursive type: I0 refers to itself" */ interface{ I0 }

	I1 /* ERROR "invalid recursive type" */ interface{ I2 }
	I2 interface{ I3 }
	I3 interface{ I1 }

	I4 interface{ f(I4) }

	// testcase for issue 5090
	I5 interface{ f(I6) }
	I6 interface{ I5 }

	// maps
	M0 map[M0 /* ERROR "invalid map key" */ ]M0

	// channels
	C0 chan C0
)

// test case for issue #34771
type (
	AA /* ERROR "invalid recursive type" */ B
	B C
	C [10]D
	D E
	E AA
)

func _() {
	type (
		t1 /* ERROR "invalid recursive type: t1 refers to itself" */ t1
		t2 *t2

		t3 t4 /* ERROR "undefined" */
		t4 t5 /* ERROR "undefined" */
		t5 t3

		// arrays
		a0 /* ERROR "invalid recursive type: a0 refers to itself" */ [10]a0
		a1 [10]*a1

		// slices
		l0 []l0

		// structs
		s0 /* ERROR "invalid recursive type: s0 refers to itself" */ struct{ _ s0 }
		s1 /* ERROR "invalid recursive type: s1 refers to itself" */ struct{ s1 }
		s2 struct{ _ *s2 }
		s3 struct{ *s3 }

		// pointers
		p0 *p0

		// functions
		f0 func(f0)
		f1 func() f1
		f2 func(f2) f2

		// interfaces
		i0 /* ERROR "invalid recursive type: i0 refers to itself" */ interface{ i0 }

		// maps
		m0 map[m0 /* ERROR "invalid map key" */ ]m0

		// channels
		c0 chan c0
	)
}

// test cases for issue 6667

type A [10]map[A /* ERROR "invalid map key" */ ]bool

type S struct {
	m map[S /* ERROR "invalid map key" */ ]bool
}

// test cases for issue 7236
// (cycle detection must not be dependent on starting point of resolution)

type (
	P1 *T9
	T9 /* ERROR "invalid recursive type: T9 refers to itself" */ T9

	T10 /* ERROR "invalid recursive type: T10 refers to itself" */ T10
	P2 *T10
)

func (T11) m() {}

type T11 /* ERROR "invalid recursive type: T11 refers to itself" */ struct{ T11 }

type T12 /* ERROR "invalid recursive type: T12 refers to itself" */ struct{ T12 }

func (*T12) m() {}

type (
	P3 *T13
	T13 /* ERROR "invalid recursive type" */ T13
)

// test cases for issue 18643
// (type cycle detection when non-type expressions are involved)
type (
	T14 /* ERROR "invalid recursive type" */ [len(T14{})]int
	T15 /* ERROR "invalid recursive type" */ [][len(T15{})]int
	T16 /* ERROR "invalid recursive type" */ map[[len(T16{1:2})]int]int
	T17 /* ERROR "invalid recursive type" */ map[int][len(T17{1:2})]int
)

// Test case for types depending on function literals (see also #22992).
type T20 chan [unsafe.Sizeof(func(ch T20){ _ = <-ch })]byte
type T22 = chan [unsafe.Sizeof(func(ch T20){ _ = <-ch })]byte

func _() {
	type T0 func(T0)
	type T1 /* ERROR "invalid recursive type" */ = func(T1)
	type T2 chan [unsafe.Sizeof(func(ch T2){ _ = <-ch })]byte
	type T3 /* ERROR "invalid recursive type" */ = chan [unsafe.Sizeof(func(ch T3){ _ = <-ch })]byte
}
