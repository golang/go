// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"time"
)

var mainTID int

func init() {
	registerInit("LockOSThreadMain", func() {
		// init is guaranteed to run on the main thread.
		mainTID = gettid()
	})
	register("LockOSThreadMain", LockOSThreadMain)

	registerInit("LockOSThreadAlt", func() {
		// Lock the OS thread now so main runs on the main thread.
		runtime.LockOSThread()
	})
	register("LockOSThreadAlt", LockOSThreadAlt)
}

func LockOSThreadMain() {
	// gettid only works on Linux, so on other platforms this just
	// checks that the runtime doesn't do anything terrible.

	// This requires GOMAXPROCS=1 from the beginning to reliably
	// start a goroutine on the main thread.
	if runtime.GOMAXPROCS(-1) != 1 {
		println("requires GOMAXPROCS=1")
		os.Exit(1)
	}

	ready := make(chan bool, 1)
	go func() {
		// Because GOMAXPROCS=1, this *should* be on the main
		// thread. Stay there.
		runtime.LockOSThread()
		if mainTID != 0 && gettid() != mainTID {
			println("failed to start goroutine on main thread")
			os.Exit(1)
		}
		// Exit with the thread locked, which should exit the
		// main thread.
		ready <- true
	}()
	<-ready
	time.Sleep(1 * time.Millisecond)
	// Check that this goroutine is still running on a different
	// thread.
	if mainTID != 0 && gettid() == mainTID {
		println("goroutine migrated to locked thread")
		os.Exit(1)
	}
	println("OK")
}

func LockOSThreadAlt() {
	// This is running locked to the main OS thread.

	var subTID int
	ready := make(chan bool, 1)
	go func() {
		// This goroutine must be running on a new thread.
		runtime.LockOSThread()
		subTID = gettid()
		ready <- true
		// Exit with the thread locked.
	}()
	<-ready
	runtime.UnlockOSThread()
	for i := 0; i < 100; i++ {
		time.Sleep(1 * time.Millisecond)
		// Check that this goroutine is running on a different thread.
		if subTID != 0 && gettid() == subTID {
			println("locked thread reused")
			os.Exit(1)
		}
		exists, supported := tidExists(subTID)
		if !supported || !exists {
			goto ok
		}
	}
	println("sub thread", subTID, "still running")
	return
ok:
	println("OK")
}
