// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// initialization cycles

package init0

// initialization cycles (we don't know the types)
const (
	s0 /* ERROR "initialization cycle: s0 refers to itself" */ = s0

	x0 /* ERROR "initialization cycle for x0" */ = y0
	y0 = x0

	a0 = b0
	b0 /* ERROR "initialization cycle for b0" */ = c0
	c0 = d0
	d0 = b0
)

var (
	s1 /* ERROR "initialization cycle: s1 refers to itself" */ = s1

	x1 /* ERROR "initialization cycle for x1" */ = y1
	y1 = x1

	a1 = b1
	b1 /* ERROR "initialization cycle for b1" */ = c1
	c1 = d1
	d1 = b1
)

// initialization cycles (we know the types)
const (
	s2 /* ERROR "initialization cycle: s2 refers to itself" */ int = s2

	x2 /* ERROR "initialization cycle for x2" */ int = y2
	y2 = x2

	a2 = b2
	b2 /* ERROR "initialization cycle for b2" */ int = c2
	c2 = d2
	d2 = b2
)

var (
	s3 /* ERROR "initialization cycle: s3 refers to itself" */ int = s3

	x3 /* ERROR "initialization cycle for x3" */ int = y3
	y3 = x3

	a3 = b3
	b3 /* ERROR "initialization cycle for b3" */ int = c3
	c3 = d3
	d3 = b3
)

// cycles via struct fields

type S1 struct {
	f int
}
const cx3 S1 /* ERROR "invalid constant type" */ = S1{cx3.f}
var vx3 /* ERROR "initialization cycle: vx3 refers to itself" */ S1 = S1{vx3.f}

// cycles via functions

var x4 = x5
var x5 /* ERROR "initialization cycle for x5" */ = f1()
func f1() int { return x5*10 }

var x6, x7 /* ERROR "initialization cycle" */ = f2()
var x8 = x7
func f2() (int, int) { return f3() + f3(), 0 }
func f3() int { return x8 }

// cycles via function literals

var x9 /* ERROR "initialization cycle: x9 refers to itself" */ = func() int { return x9 }()

var x10 /* ERROR "initialization cycle for x10" */ = f4()

func f4() int {
	_ = func() {
		_ = x10
	}
	return 0
}

// cycles via method expressions

type T1 struct{}

func (T1) m() bool { _ = x11; return false }

var x11 /* ERROR "initialization cycle for x11" */ = T1.m(T1{})

// cycles via method values

type T2 struct{}

func (T2) m() bool { _ = x12; return false }

var t1 T2
var x12 /* ERROR "initialization cycle for x12" */ = t1.m
