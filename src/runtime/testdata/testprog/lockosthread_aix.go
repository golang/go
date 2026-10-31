// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime"
)

func init() {
	registerInit("LockOSThreadExitAIX", func() {
		// Lock the OS thread now so main runs on the main thread.
		runtime.LockOSThread()
	})
	register("LockOSThreadExitAIX", LockOSThreadExitAIX)
}

// LockOSThreadExitAIX is like LockOSThreadAlt, but AIX has no thread ID
// to check without cgo. Instead, record the threads that exist while n
// goroutines are locked to their threads, and wait for n of them to
// exit. No other thread exits in this program.
func LockOSThreadExitAIX() {
	const n = 10
	ready := make(chan bool)
	done := make(chan bool)
	for range n {
		go func() {
			runtime.LockOSThread()
			ready <- true
			<-done
		}()
	}
	for range n {
		<-ready
	}
	threads := lwps()
	close(done)
	runtime.UnlockOSThread()

	// If the threads never exit, this loops until the test times out.
	for exited(threads) < n {
		runtime.Gosched()
	}
	println("OK")
}

// lwps returns the IDs of the kernel threads in the process.
func lwps() map[string]bool {
	entries, err := os.ReadDir(fmt.Sprintf("/proc/%d/lwp", os.Getpid()))
	if err != nil {
		panic(err)
	}
	ids := make(map[string]bool)
	for _, e := range entries {
		ids[e.Name()] = true
	}
	return ids
}

// exited returns how many of the given threads have exited.
func exited(threads map[string]bool) int {
	now := lwps()
	n := 0
	for id := range threads {
		if !now[id] {
			n++
		}
	}
	return n
}
