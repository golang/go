// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test behavior of the blank identifier (_).

package main

import _ "fmt"

var call string

type T struct {
	_, _, _ int
}

func (T) _() {
}

func (T) _() {
}

const (
	c0 = iota
	_
	_
	_
	c4
)

var ints = []string{
	"1",
	"2",
	"3",
}

func f() (int, int) {
	call += "f"
	return 1, 2
}

func g() (float64, float64) {
	call += "g"
	return 3, 4
}

func h(_ int, _ float64) {
}

func i() int {
	call += "i"
	return 23
}

var _ = i()

func main() {
	if call != "i" {
		panic("init did not run")
	}
	call = ""
	_, _ = f()
	a, _ := f()
	if a != 1 {
		panic(a)
	}
	b, _ := g()
	if b != 3 {
		panic(b)
	}
	_, a = f()
	if a != 2 {
		panic(a)
	}
	_, b = g()
	if b != 4 {
		panic(b)
	}
	_ = i()
	if call != "ffgfgi" {
		panic(call)
	}
	if c4 != 4 {
		panic(c4)
	}

	out := ""
	for _, s := range ints {
		out += s
	}
	if out != "123" {
		panic(out)
	}

	sum := 0
	for s := range ints {
		sum += s
	}
	if sum != 3 {
		panic(sum)
	}

	h(a, b)
	
	m()
}

type I interface {
	M(_ int, y int)
}

type TI struct{}

func (_ TI) M(x int, y int) {
	if x != y {
		println("invalid M call:", x, y)
		panic("bad M")
	}
}

var fp = func(_ int, y int) {}

func init() {
	fp = fp1
}

func fp1(x, y int) {
	if x != y {
		println("invalid fp1 call:", x, y)
		panic("bad fp1")
	}
}


func m() {
	var i I
	
	i = TI{}
	i.M(1, 1)
	i.M(2, 2)
	
	fp(1, 1)
	fp(2, 2)
}

// useless but legal
var _ int = 1
var _ = 2
var _, _ = 3, 4

const _ = 3
const _, _ = 4, 5

type _ int

func _() {
	panic("oops")
}

func ff() {
	var _ int = 1
}

