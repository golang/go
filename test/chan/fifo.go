// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that unbuffered channels act as pure fifos.

package main

import "os"

const N = 10

func AsynchFifo() {
	ch := make(chan int, N)
	for i := 0; i < N; i++ {
		ch <- i
	}
	for i := 0; i < N; i++ {
		if <-ch != i {
			print("bad receive\n")
			os.Exit(1)
		}
	}
}

func Chain(ch <-chan int, val int, in <-chan int, out chan<- int) {
	<-in
	if <-ch != val {
		panic(val)
	}
	out <- 1
}

// thread together a daisy chain to read the elements in sequence
func SynchFifo() {
	ch := make(chan int)
	in := make(chan int)
	start := in
	for i := 0; i < N; i++ {
		out := make(chan int)
		go Chain(ch, i, in, out)
		in = out
	}
	start <- 0
	for i := 0; i < N; i++ {
		ch <- i
	}
	<-in
}

func main() {
	AsynchFifo()
	SynchFifo()
}
