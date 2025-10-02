// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

// This is a set of micro-tests with obvious goroutine leaks that
// ensures goroutine leak detection works.
//
// Tests in this file are not flaky iff. run with GOMAXPROCS=1.
// The main goroutine forcefully yields via `runtime.Gosched()` before
// running the profiler. This moves them to the back of the run queue,
// allowing the leaky goroutines to be scheduled beforehand and get stuck.

func init() {
	register("NilRecv", NilRecv)
	register("NilSend", NilSend)
	register("SelectNoCases", SelectNoCases)
	register("ChanRecv", ChanRecv)
	register("ChanSend", ChanSend)
	register("Select", Select)
	register("WaitGroup", WaitGroup)
	register("MutexStack", MutexStack)
	register("MutexHeap", MutexHeap)
	register("RWMutexRLock", RWMutexRLock)
	register("RWMutexLock", RWMutexLock)
	register("Cond", Cond)
	register("Mixed", Mixed)
	register("NoLeakGlobal", NoLeakGlobal)
}

func NilRecv() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		var c chan int
		<-c
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func NilSend() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		var c chan int
		c <- 0
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func ChanRecv() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		<-make(chan int)
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func SelectNoCases() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		select {}
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func ChanSend() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		make(chan int) <- 0
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func Select() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		select {
		case make(chan int) <- 0:
		case <-make(chan int):
		}
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func WaitGroup() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		var wg sync.WaitGroup
		wg.Add(1)
		wg.Wait()
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func MutexStack() {
	prof := pprof.Lookup("goroutineleak")
	for i := 0; i < 1000; i++ {
		go func() {
			var mu sync.Mutex
			mu.Lock()
			mu.Lock()
			panic("should not be reached")
		}()
	}
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func MutexHeap() {
	prof := pprof.Lookup("goroutineleak")
	for i := 0; i < 1000; i++ {
		go func() {
			mu := &sync.Mutex{}
			go func() {
				mu.Lock()
				mu.Lock()
				panic("should not be reached")
			}()
		}()
	}
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func RWMutexRLock() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		mu := &sync.RWMutex{}
		mu.Lock()
		mu.RLock()
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func RWMutexLock() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		mu := &sync.RWMutex{}
		mu.Lock()
		mu.Lock()
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func Cond() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		cond := sync.NewCond(&sync.Mutex{})
		cond.L.Lock()
		cond.Wait()
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

func Mixed() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		ch := make(chan int)
		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			ch <- 0
			wg.Done()
			panic("should not be reached")
		}()
		wg.Wait()
		<-ch
		panic("should not be reached")
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}

var ch = make(chan int)

// No leak should be reported by this test
func NoLeakGlobal() {
	prof := pprof.Lookup("goroutineleak")
	go func() {
		<-ch
	}()
	// Yield several times to allow the child goroutine to run.
	for i := 0; i < yieldCount; i++ {
		runtime.Gosched()
	}
	prof.WriteTo(os.Stdout, 2)
}
