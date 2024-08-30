// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the behavior of closures.

package main

import "runtime"

var c = make(chan int)

func check(a []int) {
	for i := 0; i < len(a); i++ {
		n := <-c
		if n != a[i] {
			println("want", a[i], "got", n, "at", i)
			panic("fail")
		}
	}
}

func f() {
	var i, j int

	i = 1
	j = 2
	f := func() {
		c <- i
		i = 4
		g := func() {
			c <- i
			c <- j
		}
		g()
		c <- i
	}
	j = 5
	f()
}

// Accumulator generator
func accum(n int) func(int) int {
	return func(i int) int {
		n += i
		return n
	}
}

func g(a, b func(int) int) {
	c <- a(2)
	c <- b(3)
	c <- a(4)
	c <- b(5)
}

func h() {
	var x8 byte = 100
	var x64 int64 = 200

	c <- int(x8)
	c <- int(x64)
	f := func(z int) {
		g := func() {
			c <- int(x8)
			c <- int(x64)
			c <- z
		}
		g()
		c <- int(x8)
		c <- int(x64)
		c <- int(z)
	}
	x8 = 101
	x64 = 201
	f(500)
}

func newfunc() func(int) int { return func(x int) int { return x } }

func main() {
	runtime.GOMAXPROCS(1)
	var fail bool

	go f()
	check([]int{1, 4, 5, 4})

	a := accum(0)
	b := accum(1)
	go g(a, b)
	check([]int{2, 4, 6, 9})

	go h()
	check([]int{100, 200, 101, 201, 500, 101, 201, 500})

	memstats := new(runtime.MemStats)
	runtime.ReadMemStats(memstats)
	n0 := memstats.Mallocs

	x, y := newfunc(), newfunc()
	if x(1) != 1 || y(2) != 2 {
		println("newfunc returned broken funcs")
		fail = true
	}

	runtime.ReadMemStats(memstats)
	if n0 != memstats.Mallocs {
		println("newfunc allocated unexpectedly")
		fail = true
	}

	ff(1)

	if fail {
		panic("fail")
	}
}

func ff(x int) {
	call(func() {
		_ = x
	})
}

func call(func()) {
}
