// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

const spawnGCMaxDepth = 5

func init() {
	register("SpawnGC", SpawnGC)
	register("DaisyChain", DaisyChain)
}

func spawnGC(i int) {
	prof := pprof.Lookup("goroutineleak")
	if i == 0 {
		return
	}
	wg := &sync.WaitGroup{}
	wg.Add(i + 1)
	go func() {
		wg.Done()
		<-make(chan int)
	}()
	for j := 0; j < i; j++ {
		go func() {
			wg.Done()
			spawnGC(i - 1)
		}()
	}
	wg.Wait()
	runtime.Gosched()
	if i == spawnGCMaxDepth {
		prof.WriteTo(os.Stdout, 2)
	} else {
		// We want to concurrently trigger the profile in order to concurrently run
		// the GC, but we don't want to stream all the profiles to standard output.
		//
		// Only output the profile for the root call to spawnGC, and otherwise stream
		// the profile outputs to /dev/null to avoid jumbling.
		prof.WriteTo(io.Discard, 2)
	}
}

// SpawnGC spawns a tree of goroutine leaks and calls the goroutine leak profiler
// for each node in the tree. It is supposed to stress the goroutine leak profiler
// under a heavily concurrent workload.
func SpawnGC() {
	spawnGC(spawnGCMaxDepth)
}

// DaisyChain spawns a daisy-chain of runnable goroutines.
//
// Each goroutine in the chain creates a new channel and goroutine.
//
// This illustrates a pathological worstcase for the goroutine leak GC complexity,
// as opposed to the regular GC, which is not negatively affected by this pattern.
func DaisyChain() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(time.Second)
		prof.WriteTo(os.Stdout, 2)
	}()
	var chain func(i int, ch chan struct{})
	chain = func(i int, ch chan struct{}) {
		if i <= 0 {
			go func() {
				time.Sleep(time.Hour)
				ch <- struct{}{}
			}()
			return
		}
		ch2 := make(chan struct{})
		go chain(i-1, ch2)
		<-ch2
		ch <- struct{}{}
	}
	// The channel buffer avoids goroutine leaks.
	go chain(1000, make(chan struct{}, 1))
}
