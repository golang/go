// run

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func main() {
	{
		x := inuse()
		c := make(chan []byte, 10)
		c <- make([]byte, 10<<20)
		close(c)
		f1(c, x)
	}
	{
		x := inuse()
		c := make(chan []byte, 10)
		c <- make([]byte, 10<<20)
		close(c)
		f2(c, x)
	}
}

func f1(c chan []byte, start int64) {
	for x := range c {
		if delta := inuse() - start; delta < 9<<20 {
			println("BUG: f1: after alloc: expected delta at least 9MB, got: ", delta)
			println(x)
		}
		x = nil
		if delta := inuse() - start; delta > 1<<20 {
			println("BUG: f1: after alloc: expected delta below 1MB, got: ", delta)
			println(x)
		}
	}
}

func f2(c chan []byte, start int64) {
	for {
		x, ok := <-c
		if !ok {
			break
		}
		if delta := inuse() - start; delta < 9<<20 {
			println("BUG: f2: after alloc: expected delta at least 9MB, got: ", delta)
			println(x)
		}
		x = nil
		if delta := inuse() - start; delta > 1<<20 {
			println("BUG: f2: after alloc: expected delta below 1MB, got: ", delta)
			println(x)
		}
	}
}

func inuse() int64 {
	runtime.GC()
	var st runtime.MemStats
	runtime.ReadMemStats(&st)
	return int64(st.Alloc)
}
