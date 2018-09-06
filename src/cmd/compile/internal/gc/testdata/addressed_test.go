// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"testing"
)

var output string

func mypanic(t *testing.T, s string) {
	t.Fatalf(s + "\n" + output)

}

func assertEqual(t *testing.T, x, y int) {
	if x != y {
		mypanic(t, fmt.Sprintf("assertEqual failed got %d, want %d", x, y))
	}
}

func TestAddressed(t *testing.T) {
	x := f1_ssa(2, 3)
	output += fmt.Sprintln("*x is", *x)
	output += fmt.Sprintln("Gratuitously use some stack")
	output += fmt.Sprintln("*x is", *x)
	assertEqual(t, *x, 9)

	w := f3a_ssa(6)
	output += fmt.Sprintln("*w is", *w)
	output += fmt.Sprintln("Gratuitously use some stack")
	output += fmt.Sprintln("*w is", *w)
	assertEqual(t, *w, 6)

	y := f3b_ssa(12)
	output += fmt.Sprintln("*y.(*int) is", *y.(*int))
	output += fmt.Sprintln("Gratuitously use some stack")
	output += fmt.Sprintln("*y.(*int) is", *y.(*int))
	assertEqual(t, *y.(*int), 12)

	z := f3c_ssa(8)
	output += fmt.Sprintln("*z.(*int) is", *z.(*int))
	output += fmt.Sprintln("Gratuitously use some stack")
	output += fmt.Sprintln("*z.(*int) is", *z.(*int))
	assertEqual(t, *z.(*int), 8)

	args(t)
	test_autos(t)
}

//go:noinline
func f1_ssa(x, y int) *int {
	x = x*y + y
	return &x
}

//go:noinline
func f3a_ssa(x int) *int {
	return &x
}

//go:noinline
func f3b_ssa(x int) interface{} { // ./foo.go:15: internal error: f3b_ssa ~r1 (type interface {}) recorded as live on entry
	return &x
}

//go:noinline
func f3c_ssa(y int) interface{} {
	x := y
	return &x
}

type V struct {
	p    *V
	w, x int64
}

func args(t *testing.T) {
	v := V{p: nil, w: 1, x: 1}
	a := V{p: &v, w: 2, x: 2}
	b := V{p: &v, w: 0, x: 0}
	i := v.args_ssa(a, b)
	output += fmt.Sprintln("i=", i)
	assertEqual(t, int(i), 2)
}

//go:noinline
func (v V) args_ssa(a, b V) int64 {
	if v.w == 0 {
		return v.x
	}
	if v.w == 1 {
		return a.x
	}
	if v.w == 2 {
		return b.x
	}
	b.p.p = &a // v.p in caller = &a

	return -1
}

func test_autos(t *testing.T) {
	test(t, 11)
	test(t, 12)
	test(t, 13)
	test(t, 21)
	test(t, 22)
	test(t, 23)
	test(t, 31)
	test(t, 32)
}

func test(t *testing.T, which int64) {
	output += fmt.Sprintln("test", which)
	v1 := V{w: 30, x: 3, p: nil}
	v2, v3 := v1.autos_ssa(which, 10, 1, 20, 2)
	if which != v2.val() {
		output += fmt.Sprintln("Expected which=", which, "got v2.val()=", v2.val())
		mypanic(t, "Failure of expected V value")
	}
	if v2.p.val() != v3.val() {
		output += fmt.Sprintln("Expected v2.p.val()=", v2.p.val(), "got v3.val()=", v3.val())
		mypanic(t, "Failure of expected V.p value")
	}
	if which != v3.p.p.p.p.p.p.p.val() {
		output += fmt.Sprintln("Expected which=", which, "got v3.p.p.p.p.p.p.p.val()=", v3.p.p.p.p.p.p.p.val())
		mypanic(t, "Failure of expected V.p value")
	}
}

func (v V) val() int64 {
	return v.w + v.x
}

// autos_ssa uses contents of v and parameters w1, w2, x1, x2
// to initialize a bunch of locals, all of which have their
// address taken to force heap allocation, and then based on
// the value of which a pair of those locals are copied in
// various ways to the two results y, and z, which are also
// addressed. Which is expected to be one of 11-13, 21-23, 31, 32,
// and y.val() should be equal to which and y.p.val() should
// be equal to z.val().  Also, x(.p)**8 == x; that is, the
// autos are all linked into a ring.
//go:noinline
func (v V) autos_ssa(which, w1, x1, w2, x2 int64) (y, z V) {
	fill_ssa(v.w, v.x, &v, v.p) // gratuitous no-op to force addressing
	var a, b, c, d, e, f, g, h V
	fill_ssa(w1, x1, &a, &b)
	fill_ssa(w1, x2, &b, &c)
	fill_ssa(w1, v.x, &c, &d)
	fill_ssa(w2, x1, &d, &e)
	fill_ssa(w2, x2, &e, &f)
	fill_ssa(w2, v.x, &f, &g)
	fill_ssa(v.w, x1, &g, &h)
	fill_ssa(v.w, x2, &h, &a)
	switch which {
	case 11:
		y = a
		z.getsI(&b)
	case 12:
		y.gets(&b)
		z = c
	case 13:
		y.gets(&c)
		z = d
	case 21:
		y.getsI(&d)
		z.gets(&e)
	case 22:
		y = e
		z = f
	case 23:
		y.gets(&f)
		z.getsI(&g)
	case 31:
		y = g
		z.gets(&h)
	case 32:
		y.getsI(&h)
		z = a
	default:

		panic("")
	}
	return
}

// gets is an address-mentioning way of implementing
// structure assignment.
//go:noinline
func (to *V) gets(from *V) {
	*to = *from
}

// gets is an address-and-interface-mentioning way of
// implementing structure assignment.
//go:noinline
func (to *V) getsI(from interface{}) {
	*to = *from.(*V)
}

// fill_ssa initializes r with V{w:w, x:x, p:p}
//go:noinline
func fill_ssa(w, x int64, r, p *V) {
	*r = V{w: w, x: x, p: p}
}
