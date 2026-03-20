// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func main() {
	if true {
		// Disabled until CL 757343 is in.
		return
	}
	runtime.GOMAXPROCS(2)
	c := make(chan bool)
	for i := 0; i < 16; i++ {
		go func() {
			var b []byte
			for range 100000 {
				f(&b)
			}
			c <- true
		}()
	}
	for i := 0; i < 16; i++ {
		<-c
	}
}

var n int = 64 // constant, but the compiler doesn't know that

//go:noinline
func f(sink *[]byte) {
	useStack(64) // Use 64KB of stack, so that shrinking might happen below.

	x := make([]int, n, 128)             // on stack
	_ = append(x, make([]int, 128-n)...) // memclrNoHeapPointersPreemptible call is here

	*sink = make([]byte, 1024) // make some garbage to cause GC
}

//go:noinline
func useStack(depth int) {
	var b [128]int
	if depth == b[depth%len(b)] { // depth == 0
		return
	}
	useStack(depth - 1)
}
