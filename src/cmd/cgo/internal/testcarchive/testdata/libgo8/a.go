// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

import (
	"os"
	"runtime"
	"sync/atomic"
)

var started, stop int32

var sink uint32

var done = make(chan bool)

// Start a goroutine that loops until it is told to stop.
// The inner loop(s) of spin contain no preemption checks;
// preemption must come from a signal handler (which should be
// installed even when coming from cgo).
func init() {
	runtime.GOMAXPROCS(1)
	go func() {
		spin()
		done <- true
	}()
}

// spin loops without calls. With GOMAXPROCS(1), GoFunction8 can only run
// and tell it to stop after asynchronous preemption. spin is a leaf
// function, which keeps its return address in the link register on arm64,
// so its return checks the link register that preemption restored.
// The nested loops and sink add preemption targets.
//
//go:noinline
func spin() {
	const K = 2_000_000_000
	for i := 0; i < K; i++ {
		for i := 0; i < K; i++ {
			for i := 0; i < K; i++ {
				// atomic.StoreInt32 is intrinsic or asm-nosplit, no preemption check.
				atomic.StoreInt32(&started, 1)
				sink++
				if atomic.LoadInt32(&stop) != 0 {
					return
				}
			}
		}
	}
}

//export GoFunction8
func GoFunction8() {
	for atomic.LoadInt32(&started) == 0 {
		runtime.Gosched()
	}
	atomic.StoreInt32(&stop, 1)
	<-done
	os.Exit(0)
}

func main() {
}
