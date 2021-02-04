// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// assignment order in multiple assignments.
// See issue #23017

package main

import "fmt"

func main() {}

func init() {
	var m = map[int]int{}
	var p *int

	defer func() {
		recover()
		check(1, len(m))
		check(42, m[2])
	}()
	m[2], *p = 42, 2
}

func init() {
	var m = map[int]int{}
	p := []int{}

	defer func() {
		recover()
		check(1, len(m))
		check(2, m[2])
	}()
	m[2], p[1] = 2, 2
}

func init() {
	type P struct{ i int }
	var m = map[int]int{}
	var p *P

	defer func() {
		recover()
		check(1, len(m))
		check(3, m[2])
	}()
	m[2], p.i = 3, 2
}

func init() {
	type T struct{ i int }
	var x T
	p := &x
	p, p.i = new(T), 4
	check(4, x.i)
}

func init() {
	var m map[int]int
	var a int
	var p = &a

	defer func() {
		recover()
		check(5, *p)
	}()
	*p, m[2] = 5, 2
}

var g int

func init() {
	var m map[int]int
	defer func() {
		recover()
		check(0, g)
	}()
	m[0], g = 1, 2
}

func init() {
	type T struct{ x struct{ y int } }
	var x T
	p := &x
	p, p.x.y = new(T), 7
	check(7, x.x.y)
	check(0, p.x.y)
}

func init() {
	type T *struct{ x struct{ y int } }
	x := struct{ y int }{0}
	var q T = &struct{ x struct{ y int } }{x}
	p := q
	p, p.x.y = nil, 7
	check(7, q.x.y)
}

func init() {
	x, y := 1, 2
	x, y = y, x
	check(2, x)
	check(1, y)
}

func check(want, got int) {
	if want != got {
		panic(fmt.Sprintf("wanted %d, but got %d", want, got))
	}
}
