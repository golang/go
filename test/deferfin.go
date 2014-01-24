// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that defers do not prevent garbage collection.

package main

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

var sink func()

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
	N := 10
	count := int32(N)
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			v := new(string)
			f := func() {
				if *v != "" {
					panic("oops")
				}
			}
			if *v != "" {
				// let the compiler think f escapes
				sink = f
			}
			runtime.SetFinalizer(v, func(p *string) {
				atomic.AddInt32(&count, -1)
			})
			defer f()
		}()
	}
	wg.Wait()
	for i := 0; i < 3; i++ {
		time.Sleep(10 * time.Millisecond)
		runtime.GC()
	}
	if count != 0 {
		println(count, "out of", N, "finalizer are not called")
		panic("not all finalizers are called")
	}
}

