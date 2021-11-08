// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var ch chan int
var x int

func f() int {
	close(ch)
	ch = nil
	return 0
}

func g() int {
	ch = nil
	x = 0
	return 0
}

func main() {
	var nilch chan int
	var v int
	var ok bool
	_, _ = v, ok

	ch = make(chan int)
	select {
	case <-ch:
	case nilch <- f():
	}

	ch = make(chan int)
	select {
	case v = <-ch:
	case nilch <- f():
	}

	ch = make(chan int)
	select {
	case v := <-ch: _ = v
	case nilch <- f():
	}

	ch = make(chan int)
	select {
	case v, ok = <-ch:
	case nilch <- f():
	}

	ch = make(chan int)
	select {
	case v, ok := <-ch: _, _ = v, ok
	case nilch <- f():
	}

	ch1 := make(chan int, 1)
	ch = ch1
	x = 42
	select {
	case ch <- x:
	case nilch <- g():
	}
	if got := <-ch1; got != 42 {
		panic(got)
	}
}
