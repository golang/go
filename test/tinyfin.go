// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test finalizers work for tiny (combined) allocations.

package main

import (
	"runtime"
	"time"
)

func main() {
	// Does not work on gccgo due to partially conservative GC.
	// Try to enable when we have fully precise GC.
	if runtime.Compiler == "gccgo" {
		return
	}
	const N = 100
	finalized := make(chan int32, N)
	for i := 0; i < N; i++ {
		x := new(int32) // subject to tiny alloc
		*x = int32(i)
		// the closure must be big enough to be combined
		runtime.SetFinalizer(x, func(p *int32) {
			finalized <- *p
		})
	}
	runtime.GC()
	count := 0
	done := make([]bool, N)
	timeout := time.After(5*time.Second)
	for {
		select {
		case <-timeout:
			println("timeout,", count, "finalized so far")
			panic("not all finalizers are called")
		case x := <-finalized:
			// Check that p points to the correct subobject of the tiny allocation.
			// It's a bit tricky, because we can't capture another variable
			// with the expected value (it would be combined as well).
			if x < 0 || x >= N {
				println("got", x)
				panic("corrupted")
			}
			if done[x] {
				println("got", x)
				panic("already finalized")
			}
			done[x] = true
			count++
			if count > N/10*9 {
				// Some of the finalizers may not be executed,
				// if the outermost allocations are combined with something persistent.
				// Currently 4 int32's are combined into a 16-byte block,
				// ensure that most of them are finalized.
				return
			}
		}
	}
}
