// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Order of operations in select.

package main

func main() {
	c := make(chan int, 1)
	x := 0
	select {
	case c <- x: // should see x = 0, not x = 42 (after makec)
	case <-makec(&x): // should be evaluated only after c and x on previous line
	}
	y := <-c
	if y != 0 {
		panic(y)
	}
}

func makec(px *int) chan bool {
	if false { for {} }
	*px = 42
	return make(chan bool, 0)
}
