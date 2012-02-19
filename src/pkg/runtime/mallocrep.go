// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Repeated malloc test.

// +build ignore

package main

import (
	"flag"
	"runtime"
)

var chatty = flag.Bool("v", false, "chatty")

var oldsys uint64
var memstats runtime.MemStats

func bigger() {
	st := &memstats
	runtime.ReadMemStats(st)
	if oldsys < st.Sys {
		oldsys = st.Sys
		if *chatty {
			println(st.Sys, " system bytes for ", st.Alloc, " Go bytes")
		}
		if st.Sys > 1e9 {
			println("too big")
			panic("fail")
		}
	}
}

func main() {
	runtime.GC()                    // clean up garbage from init
	runtime.ReadMemStats(&memstats) // first call can do some allocations
	runtime.MemProfileRate = 0      // disable profiler
	stacks := memstats.Alloc        // ignore stacks
	flag.Parse()
	for i := 0; i < 1<<7; i++ {
		for j := 1; j <= 1<<22; j <<= 1 {
			if i == 0 && *chatty {
				println("First alloc:", j)
			}
			if a := memstats.Alloc - stacks; a != 0 {
				println("no allocations but stats report", a, "bytes allocated")
				panic("fail")
			}
			b := runtime.Alloc(uintptr(j))
			runtime.ReadMemStats(&memstats)
			during := memstats.Alloc - stacks
			runtime.Free(b)
			runtime.ReadMemStats(&memstats)
			if a := memstats.Alloc - stacks; a != 0 {
				println("allocated ", j, ": wrong stats: during=", during, " after=", a, " (want 0)")
				panic("fail")
			}
			bigger()
		}
		if i%(1<<10) == 0 && *chatty {
			println(i)
		}
		if i == 0 {
			if *chatty {
				println("Primed", i)
			}
			//	runtime.frozen = true
		}
	}
}
