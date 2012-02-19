// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test functions and goroutines.

package main

func caller(f func(int, int) int, a, b int, c chan int) {
	c <- f(a, b)
}

func gocall(f func(int, int) int, a, b int) int {
	c := make(chan int)
	go caller(f, a, b, c)
	return <-c
}

func call(f func(int, int) int, a, b int) int {
	return f(a, b)
}

func call1(f func(int, int) int, a, b int) int {
	return call(f, a, b)
}

var f func(int, int) int

func add(x, y int) int {
	return x + y
}

func fn() func(int, int) int {
	return f
}

var fc func(int, int, chan int)

func addc(x, y int, c chan int) {
	c <- x+y
}

func fnc() func(int, int, chan int) {
	return fc
}

func three(x int) {
	if x != 3 {
		println("wrong val", x)
		panic("fail")
	}
}

var notmain func()

func emptyresults() {}
func noresults()    {}

var nothing func()

func main() {
	three(call(add, 1, 2))
	three(call1(add, 1, 2))
	f = add
	three(call(f, 1, 2))
	three(call1(f, 1, 2))
	three(call(fn(), 1, 2))
	three(call1(fn(), 1, 2))
	three(call(func(a, b int) int { return a + b }, 1, 2))
	three(call1(func(a, b int) int { return a + b }, 1, 2))

	fc = addc
	c := make(chan int)
	go addc(1, 2, c)
	three(<-c)
	go fc(1, 2, c)
	three(<-c)
	go fnc()(1, 2, c)
	three(<-c)
	go func(a, b int, c chan int) { c <- a+b }(1, 2, c)
	three(<-c)

	emptyresults()
	noresults()
	nothing = emptyresults
	nothing()
	nothing = noresults
	nothing()
}
