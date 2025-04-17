// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file

// Test that a select statement proceeds when a value is ready.

package main

func f() *int {
	println("BUG: called f")
	return new(int)
}

func main() {
	var x struct {
		a int
	}
	c := make(chan int, 1)
	c1 := make(chan int)
	c <- 42
	select {
	case *f() = <-c1:
		// nothing
	case x.a = <-c:
		if x.a != 42 {
			println("BUG:", x.a)
		}
	}
}
