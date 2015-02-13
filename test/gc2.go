// +build !nacl
// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that buffered channels are garbage collected properly.
// An interesting case because they have finalizers and used to
// have self loops that kept them from being collected.
// (Cyclic data with finalizers is never finalized, nor collected.)

package main

import (
	"fmt"
	"os"
	"runtime"
)

func main() {
	const N = 10000
	st := new(runtime.MemStats)
	memstats := new(runtime.MemStats)
	runtime.ReadMemStats(st)
	for i := 0; i < N; i++ {
		c := make(chan int, 10)
		_ = c
		if i%100 == 0 {
			for j := 0; j < 4; j++ {
				runtime.GC()
				runtime.Gosched()
				runtime.GC()
				runtime.Gosched()
			}
		}
	}

	runtime.ReadMemStats(memstats)
	obj := int64(memstats.HeapObjects - st.HeapObjects)
	if obj > N/5 {
		fmt.Println("too many objects left:", obj)
		os.Exit(1)
	}
}
