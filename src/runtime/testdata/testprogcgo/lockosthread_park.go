// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func init() {
	register("LockOSThreadParkExit", LockOSThreadParkExit)
}

// LockOSThreadParkExit repeatedly starts a goroutine that locks its
// thread, parks once, and returns while still locked, so the runtime
// exits the thread. A parked M owns an OS semaphore on some systems.
// On AIX, sem_init fails once SEM_NSEMS_MAX semaphores exist (32768
// on AIX 7.3), so this fails there if the runtime does not release
// the semaphore when the thread exits.
func LockOSThreadParkExit() {
	const n = 40000
	for range n {
		done := make(chan bool)
		go func() {
			runtime.LockOSThread()
			runtime.Gosched()
			close(done)
		}()
		<-done
	}
	println("OK")
}
