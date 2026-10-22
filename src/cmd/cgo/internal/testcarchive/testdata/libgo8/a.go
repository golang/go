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

var started int32

var sink uint32

// Start a goroutine that loops "forever".
// the inner loop(s) of the subroutine contains
// no preemption checks; preemption must come
// from a signal handler (which should be
// installed even when coming from cgo).
func init() {
	runtime.GOMAXPROCS(1)
	go func() {
		const K = 2_000_000_000
		for i := 0; i < K; i++ {
			for i := 0; i < K; i++ {
				for i := 0; i < K; i++ {
					// atomic.StoreInt32 is intrinsic or asm-nosplit, no preemption check.
					atomic.StoreInt32(&started, 1)
					sink++
				}
			}
		}
	}()
}

//export GoFunction8
func GoFunction8() {
	for atomic.LoadInt32(&started) == 0 {
		runtime.Gosched()
	}
	os.Exit(0)
}

func main() {
}
