// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/syscall/unix"
	"runtime"
)

func init() {
	register("LockOSThreadVgetrandom", LockOSThreadVgetrandom)
}

var sinkInt int

func LockOSThreadVgetrandom() {
	// This is a regression test for https://go.dev/issue/73141. When that
	// reproduces, this crashes with SIGSEGV with no output or stack trace,
	// and detail only available in a core file.
	//
	// Thread exit via mexit cleans up vgetrandom state. Stress test thread
	// exit + vgetrandom to look for issues by creating lots of threads
	// that use GetRandom and then exit.

	// Launch at most 100 threads at a time.
	const parallelism = 100
	ch := make(chan struct{}, parallelism)
	for range 100 {
		ch <- struct{}{}
	}

	// Create at most 1000 threads to avoid completely exhausting the
	// system. This test generally reproduces https://go.dev/issue/73141 in
	// less than 500 iterations.
	const iterations = 1000
	for range iterations {
		<-ch
		go func() {
			defer func() {
				ch <- struct{}{}
			}()

			// Exit with LockOSThread held.
			runtime.LockOSThread()

			// Be sure to use GetRandom to initialize vgetrandom state.
			b := make([]byte, 1)
			_, err := unix.GetRandom(b, 0)
			if err != nil {
				panic(err)
			}

			// Do some busy-work. It is unclear why this is
			// necessary to reproduce. Perhaps to introduce
			// interesting scheduling where threads get descheduled
			// in the middle of getting or putting vgetrandom
			// state.
			for range 10 * 1000 * 1000 {
				sinkInt = 1
			}
		}()
	}

	// Wait for all threads to finish.
	for range parallelism {
		<-ch
	}
	println("OK")
}
