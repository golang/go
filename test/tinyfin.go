// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test finalizers work for tiny (combined) allocations.

package main

import (
	"runtime"
	"sync/atomic"
	"time"
)

func main() {
	// Does not work on 32-bits due to partially conservative GC.
	// Try to enable when we have fully precise GC.
	if runtime.GOARCH != "amd64" {
		return
	}
	// Likewise for gccgo.
	if runtime.Compiler == "gccgo" {
		return
	}
	N := int32(100)
	count := N
	done := make([]bool, N)
	for i := int32(0); i < N; i++ {
		x := i // subject to tiny alloc
		// the closure must be big enough to be combined
		runtime.SetFinalizer(&x, func(p *int32) {
			// Check that p points to the correct subobject of the tiny allocation.
			// It's a bit tricky, because we can't capture another variable
			// with the expected value (it would be combined as well).
			if *p < 0 || *p >= N {
				println("got", *p)
				panic("corrupted")
			}
			if done[*p] {
				println("got", *p)
				panic("already finalized")
			}
			done[*p] = true
			atomic.AddInt32(&count, -1)
		})
	}
	for i := 0; i < 4; i++ {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	// Some of the finalizers may not be executed,
	// if the outermost allocations are combined with something persistent.
	// Currently 4 int32's are combined into a 16-byte block,
	// ensure that most of them are finalized.
	if count >= N/4 {
		println(count, "out of", N, "finalizer are not called")
		panic("not all finalizers are called")
	}
}

