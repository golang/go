// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

const N = 10

var count int64

func run() error {
	f1 := func() {}
	f2 := func() {
		func() {
			f1()
		}()
	}
	runtime.SetFinalizer(&f1, func(f *func()) {
		atomic.AddInt64(&count, -1)
	})
	go f2()
	return nil
}

func main() {
	// Does not work with gccgo, due to partially conservative GC.
	// Try to enable when we have fully precise GC.
	if runtime.Compiler == "gccgo" {
		return
	}
	count = N
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			run()
			wg.Done()
		}()
	}
	wg.Wait()
	for i := 0; i < 2*N; i++ {
		time.Sleep(10 * time.Millisecond)
		runtime.GC()
	}
	if count != 0 {
		println(count, "out of", N, "finalizer are not called")
		panic("not all finalizers are called")
	}
}
