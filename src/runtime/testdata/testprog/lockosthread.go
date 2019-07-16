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

	registerInit("LockOSThreadAvoidsStatePropagation", func() {
		// Lock the OS thread now so main runs on the main thread.
		runtime.LockOSThread()
	})
	register("LockOSThreadAvoidsStatePropagation", LockOSThreadAvoidsStatePropagation)
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

func LockOSThreadAvoidsStatePropagation() {
	// This test is similar to LockOSThreadAlt in that it will detect if a thread
	// which should have died is still running. However, rather than do this with
	// thread IDs, it does this by unsharing state on that thread. This way, it
	// also detects whether new threads were cloned from the dead thread, and not
	// from a clean thread. Cloning from a locked thread is undesirable since
	// cloned threads will inherit potentially unwanted OS state.
	//
	// unshareFs, getcwd, and chdir("/tmp") are only guaranteed to work on
	// Linux, so on other platforms this just checks that the runtime doesn't
	// do anything terrible.
	//
	// This is running locked to the main OS thread.

	// GOMAXPROCS=1 makes this fail much more reliably if a tainted thread is
	// cloned from.
	if runtime.GOMAXPROCS(-1) != 1 {
		println("requires GOMAXPROCS=1")
		os.Exit(1)
	}

	if err := chdir("/"); err != nil {
		println("failed to chdir:", err.Error())
		os.Exit(1)
	}
	// On systems other than Linux, cwd == "".
	cwd, err := getcwd()
	if err != nil {
		println("failed to get cwd:", err.Error())
		os.Exit(1)
	}
	if cwd != "" && cwd != "/" {
		println("unexpected cwd", cwd, " wanted /")
		os.Exit(1)
	}

	ready := make(chan bool, 1)
	go func() {
		// This goroutine must be running on a new thread.
		runtime.LockOSThread()

		// Unshare details about the FS, like the CWD, with
		// the rest of the process on this thread.
		// On systems other than Linux, this is a no-op.
		if err := unshareFs(); err != nil {
			if err == errNotPermitted {
				println("unshare not permitted")
				os.Exit(0)
			}
			println("failed to unshare fs:", err.Error())
			os.Exit(1)
		}
		// Chdir to somewhere else on this thread.
		// On systems other than Linux, this is a no-op.
		if err := chdir("/tmp"); err != nil {
			println("failed to chdir:", err.Error())
			os.Exit(1)
		}

		// The state on this thread is now considered "tainted", but it
		// should no longer be observable in any other context.

		ready <- true
		// Exit with the thread locked.
	}()
	<-ready

	// Spawn yet another goroutine and lock it. Since GOMAXPROCS=1, if
	// for some reason state from the (hopefully dead) locked thread above
	// propagated into a newly created thread (via clone), or that thread
	// is actually being re-used, then we should get scheduled on such a
	// thread with high likelihood.
	done := make(chan bool)
	go func() {
		runtime.LockOSThread()

		// Get the CWD and check if this is the same as the main thread's
		// CWD. Every thread should share the same CWD.
		// On systems other than Linux, wd == "".
		wd, err := getcwd()
		if err != nil {
			println("failed to get cwd:", err.Error())
			os.Exit(1)
		}
		if wd != cwd {
			println("bad state from old thread propagated after it should have died")
			os.Exit(1)
		}
		<-done

		runtime.UnlockOSThread()
	}()
	done <- true
	runtime.UnlockOSThread()
	println("OK")
}
