// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test reordering of assignments.

package main

import "fmt"

func main() {
	p1()
	p2()
	p3()
	p4()
	p5()
	p6()
	p7()
	p8()
	p9()
	p10()
	p11()
}

var gx []int

func f(i int) int {
	return gx[i]
}

func check(x []int, x0, x1, x2 int) {
	if x[0] != x0 || x[1] != x1 || x[2] != x2 {
		fmt.Printf("%v, want %d,%d,%d\n", x, x0, x1, x2)
		panic("failed")
	}
}

func check3(x, y, z, xx, yy, zz int) {
	if x != xx || y != yy || z != zz {
		fmt.Printf("%d,%d,%d, want %d,%d,%d\n", x, y, z, xx, yy, zz)
		panic("failed")
	}
}

func p1() {
	x := []int{1, 2, 3}
	i := 0
	i, x[i] = 1, 100
	_ = i
	check(x, 100, 2, 3)
}

func p2() {
	x := []int{1, 2, 3}
	i := 0
	x[i], i = 100, 1
	_ = i
	check(x, 100, 2, 3)
}

func p3() {
	x := []int{1, 2, 3}
	y := x
	gx = x
	x[1], y[0] = f(0), f(1)
	check(x, 2, 1, 3)
}

func p4() {
	x := []int{1, 2, 3}
	y := x
	gx = x
	x[1], y[0] = gx[0], gx[1]
	check(x, 2, 1, 3)
}

func p5() {
	x := []int{1, 2, 3}
	y := x
	p := &x[0]
	q := &x[1]
	*p, *q = x[1], y[0]
	check(x, 2, 1, 3)
}

func p6() {
	x := 1
	y := 2
	z := 3
	px := &x
	py := &y
	*px, *py = y, x
	check3(x, y, z, 2, 1, 3)
}

func f1(x, y, z int) (xx, yy, zz int) {
	return x, y, z
}

func f2() (x, y, z int) {
	return f1(2, 1, 3)
}

func p7() {
	x, y, z := f2()
	check3(x, y, z, 2, 1, 3)
}

func p8() {
	m := make(map[int]int)
	m[0] = len(m)
	if m[0] != 0 {
		panic(m[0])
	}
}

// Issue #13433: Left-to-right assignment of OAS2XXX nodes.
func p9() {
	var x bool

	// OAS2FUNC
	x, x = fn()
	checkOAS2XXX(x, "x, x = fn()")

	// OAS2RECV
	var c = make(chan bool, 10)
	c <- false
	x, x = <-c
	checkOAS2XXX(x, "x, x <-c")

	// OAS2MAPR
	var m = map[int]bool{0: false}
	x, x = m[0]
	checkOAS2XXX(x, "x, x = m[0]")

	// OAS2DOTTYPE
	var i interface{} = false
	x, x = i.(bool)
	checkOAS2XXX(x, "x, x = i.(bool)")
}

//go:noinline
func fn() (bool, bool) { return false, true }

// checks the order of OAS2XXX.
func checkOAS2XXX(x bool, s string) {
	if !x {
		fmt.Printf("%s; got=(false); want=(true)\n", s)
		panic("failed")
	}
}

//go:noinline
func fp() (*int, int) { return nil, 42 }

func p10() {
	p := new(int)
	p, *p = fp()
}

func p11() {
	var i interface{}
	p := new(bool)
	p, *p = i.(*bool)
}
