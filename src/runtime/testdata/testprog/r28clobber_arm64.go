// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && arm64

package main

import (
	"os"
	"runtime"
	"sync"
	"syscall"
)

func init() {
	register("R28ClobberSIGURG", R28ClobberSIGURG)
	register("R28ClobberSIGURGClone", R28ClobberSIGURGClone)
}

// Declared in r28clobber_arm64.s
func r28DirtyLoop(tid, pid int, sig, iters int32)

func R28ClobberSIGURG() {
	runtime.LockOSThread()
	r28DirtyLoop(syscall.Gettid(), syscall.Getpid(), int32(syscall.SIGURG), 100)
	os.Exit(0)
}

func R28ClobberSIGURGClone() {
	// Lock the main thread so the new goroutine must run on a fresh OS thread.
	runtime.LockOSThread()
	runtime.GOMAXPROCS(2)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		runtime.LockOSThread()
		r28DirtyLoop(syscall.Gettid(), syscall.Getpid(), int32(syscall.SIGURG), 100)
	}()
	wg.Wait()
	os.Exit(0)
}
