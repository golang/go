// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test various parsing cases that are a little
// different now that send is a statement, not an expression.

package main

func main() {
	chanchan()
	sendprec()
}

func chanchan() {
	cc := make(chan chan int, 1)
	c := make(chan int, 1)
	cc <- c
	select {
	case <-cc <- 2:
	default:
		panic("nonblock")
	}
	if <-c != 2 {
		panic("bad receive")
	}
}

func sendprec() {
	c := make(chan bool, 1)
	c <- false || true // not a syntax error: same as c <- (false || true)
	if !<-c {
		panic("sent false")
	}
}
