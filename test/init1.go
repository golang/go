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

	const chunk = 1 << 20
	memstats := new(runtime.MemStats)
	runtime.ReadMemStats(memstats)
	sys := memstats.Sys
	b := make([]byte, chunk)
	for i := range b {
		b[i] = byte(i%10 + '0')
	}
	s := string(b)
	for i := 0; i < 1000; i++ {
		x = []byte(s)
	}
	runtime.ReadMemStats(memstats)
	sys1 := memstats.Sys
	if sys1-sys > chunk*50 {
		println("allocated 1000 chunks of", chunk, "and used ", sys1-sys, "memory")
		panic("init1")
	}
}

func send(c chan int) {
	c <- 1
}

func main() {
}
