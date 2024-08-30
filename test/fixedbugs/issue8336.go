// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8336. Order of evaluation of receive channels in select.

package main

type X struct {
	c chan int
}

func main() {
	defer func() {
		recover()
	}()
	var x *X
	select {
	case <-x.c: // should fault and panic before foo is called
	case <-foo():
	}
}

func foo() chan int {
	println("BUG: foo must not be called")
	return make(chan int)
}
