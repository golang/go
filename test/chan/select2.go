// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func sender(c chan int, n int) {
	for i := 0; i < n; i++ {
		c <- 1
	}
}

func receiver(c, dummy chan int, n int) {
	for i := 0; i < n; i++ {
		select {
		case <-c:
			// nothing
		case <-dummy:
			panic("dummy")
		}
	}
}

func main() {
	runtime.MemProfileRate = 0

	c := make(chan int)
	dummy := make(chan int)

	// warm up
	go sender(c, 100000)
	receiver(c, dummy, 100000)
	runtime.GC()
	runtime.MemStats.Alloc = 0

	// second time shouldn't increase footprint by much
	go sender(c, 100000)
	receiver(c, dummy, 100000)
	runtime.GC()

	if runtime.MemStats.Alloc > 1e5 {
		println("BUG: too much memory for 100,000 selects:", runtime.MemStats.Alloc)
	}
}
