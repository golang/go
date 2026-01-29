// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build test_run

// Compute Fibonacci numbers with two goroutines
// that pass integers back and forth. No actual
// concurrency, just threads and synchronization
// and foreign code on multiple pthreads.

package main

import (
	"runtime"
	"strconv"

	"cgostdio/stdio"
)

func fibber(c, out chan int64, i int64) {
	// Keep the fibbers in dedicated operating system
	// threads, so that this program tests coordination
	// between pthreads and not just goroutines.
	runtime.LockOSThread()

	if i == 0 {
		c <- i
	}
	for {
		j := <-c
		stdio.Stdout.WriteString(strconv.FormatInt(j, 10) + "\n")
		out <- j
		<-out
		i += j
		c <- i
	}
}

func main() {
	c := make(chan int64)
	out := make(chan int64)
	go fibber(c, out, 0)
	go fibber(c, out, 1)
	<-out
	for i := 0; i < 90; i++ {
		out <- 1
		<-out
	}
}
