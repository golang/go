// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// type declarations

package P0

type (
	B bool
	I int32
	A [10]P
	T struct {
		x, y P
	}
	P *T
	R *R
	F func(A) I
	Y interface {
		f(A) I
	}
	S []P
	M map[I]F
	C chan<- I
)

type (
	a/* ERROR "illegal cycle" */ a
	a/* ERROR "already declared" */ int

	b/* ERROR "illegal cycle" */ c
	c d
	d e
	e b /* ERROR "not a type" */

	t *t

	U V
	V W
	W *U

	P1 *S2
	P2 P1

	S1 struct {
		a, b, c int
		u, v, a/* ERROR "already declared" */ float
	}
	S2/* ERROR "illegal cycle" */ struct {
		x S2
	}

	L1 []L1
	L2 []int

	A1 [10]int
	A2/* ERROR "illegal cycle" */ [10]A2
	A3/* ERROR "illegal cycle" */ [10]struct {
		x A4
	}
	A4 [10]A3

	F1 func()
	F2 func(x, y, z float)
	F3 func(x, y, x /* ERROR "already declared" */ float)
	F4 func() (x, y, x /* ERROR "already declared" */ float)
	F5 func(x int) (x /* ERROR "already declared" */ float)

	I1 interface{}
	I2 interface {
		m1()
	}
	I3 interface {
		m1()
		m1 /* ERROR "already declared" */ ()
	}
	I4 interface {
		m1(x, y, x /* ERROR "already declared" */ float)
		m2() (x, y, x /* ERROR "already declared" */ float)
		m3(x int) (x /* ERROR "already declared" */ float)
	}
	I5 interface {
		m1(I5)
	}

	C1 chan int
	C2 <-chan int
	C3 chan<- C3

	M1 map[Last]string
	M2 map[string]M2

	Last int
)
