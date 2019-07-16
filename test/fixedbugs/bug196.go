// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var m = map[int]int{0: 0, 1: 0}
var nf = 0
var i int

func multi() (int, int) { return 1, 2 }

func xxx() {
	var c chan int
	x, ok := <-c

	var m map[int]int
	x, ok = m[1]

	var i interface{}
	var xx int
	xx, ok = i.(int)

	a, b := multi()

	_, _, _, _, _ = x, ok, xx, a, b
}

func f() map[int]int {
	nf++
	return m
}

func g() *int {
	nf++
	return &i
}

func main() {
	f()[0]++
	f()[1] += 2
	*g() %= 2
	if nf != 3 {
		println("too many calls:", nf)
		panic("fail")
	}

}
