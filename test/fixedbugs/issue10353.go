// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 10253: cmd/gc: incorrect escape analysis of closures
// Partial call x.foo was not promoted to heap.

package main

func main() {
	c := make(chan bool)
	// Create a new goroutine to get a default-size stack segment.
	go func() {
		x := new(X)
		clos(x.foo)()
		c <- true
	}()
	<-c
}

type X int

func (x *X) foo() {
}

func clos(x func()) func() {
	f := func() {
		print("")
		x() // This statement crashed, because the partial call was allocated on the old stack.
	}
	// Grow stack so that partial call x becomes invalid if allocated on stack.
	growstack(10000)
	c := make(chan bool)
	// Spoil the previous stack segment.
	go func() {
		c <- true
	}()
	<-c
	return f
}

func growstack(x int) {
	if x == 0 {
		return
	}
	growstack(x - 1)
}
