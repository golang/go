// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that goroutines and garbage collection run during init.

package main

import "runtime"

var x []byte

func init() {
	c := make(chan int)
	go send(c)
	<-c

	const N = 1000
	const MB = 1 << 20
	b := make([]byte, MB)
	for i := range b {
		b[i] = byte(i%10 + '0')
	}
	s := string(b)

	memstats := new(runtime.MemStats)
	runtime.ReadMemStats(memstats)
	sys, numGC := memstats.Sys, memstats.NumGC

	// Generate 1,000 MB of garbage, only retaining 1 MB total.
	for i := 0; i < N; i++ {
		x = []byte(s)
	}

	// Verify that the garbage collector ran by seeing if we
	// allocated fewer than N*MB bytes from the system.
	runtime.ReadMemStats(memstats)
	sys1, numGC1 := memstats.Sys, memstats.NumGC
	if sys1-sys >= N*MB || numGC1 == numGC {
		println("allocated 1000 chunks of", MB, "and used ", sys1-sys, "memory")
		println("numGC went", numGC, "to", numGC)
		panic("init1")
	}
}

func send(c chan int) {
	c <- 1
}

func main() {
}
