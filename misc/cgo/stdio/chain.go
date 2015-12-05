// cmpout

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build test_run

// Pass numbers along a chain of threads.

package main

import (
	"runtime"
	"strconv"

	"../stdio"
)

const N = 10
const R = 5

func link(left chan<- int, right <-chan int) {
	// Keep the links in dedicated operating system
	// threads, so that this program tests coordination
	// between pthreads and not just goroutines.
	runtime.LockOSThread()
	for {
		v := <-right
		stdio.Stdout.WriteString(strconv.Itoa(v) + "\n")
		left <- 1 + v
	}
}

func main() {
	leftmost := make(chan int)
	var left chan int
	right := leftmost
	for i := 0; i < N; i++ {
		left, right = right, make(chan int)
		go link(left, right)
	}
	for i := 0; i < R; i++ {
		right <- 0
		x := <-leftmost
		stdio.Stdout.WriteString(strconv.Itoa(x) + "\n")
	}
}
