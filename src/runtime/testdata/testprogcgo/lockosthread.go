// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

package main

import (
	"os"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"
)

/*
#include <pthread.h>
#include <stdint.h>

extern uint32_t threadExited;

void setExited(void *x);
*/
import "C"

var mainThread C.pthread_t

func init() {
	registerInit("LockOSThreadMain", func() {
		// init is guaranteed to run on the main thread.
		mainThread = C.pthread_self()
	})
	register("LockOSThreadMain", LockOSThreadMain)

	registerInit("LockOSThreadAlt", func() {
		// Lock the OS thread now so main runs on the main thread.
		runtime.LockOSThread()
	})
	register("LockOSThreadAlt", LockOSThreadAlt)
}

func LockOSThreadMain() {
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
		self := C.pthread_self()
		if C.pthread_equal(mainThread, self) == 0 {
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
	self := C.pthread_self()
	if C.pthread_equal(mainThread, self) != 0 {
		println("goroutine migrated to locked thread")
		os.Exit(1)
	}
	println("OK")
}

func LockOSThreadAlt() {
	// This is running locked to the main OS thread.

	var subThread C.pthread_t
	ready := make(chan bool, 1)
	C.threadExited = 0
	go func() {
		// This goroutine must be running on a new thread.
		runtime.LockOSThread()
		subThread = C.pthread_self()
		// Register a pthread destructor so we can tell this
		// thread has exited.
		var key C.pthread_key_t
		C.pthread_key_create(&key, (*[0]byte)(unsafe.Pointer(C.setExited)))
		C.pthread_setspecific(key, unsafe.Pointer(new(int)))
		ready <- true
		// Exit with the thread locked.
	}()
	<-ready
	for {
		time.Sleep(1 * time.Millisecond)
		// Check that this goroutine is running on a different thread.
		self := C.pthread_self()
		if C.pthread_equal(subThread, self) != 0 {
			println("locked thread reused")
			os.Exit(1)
		}
		if atomic.LoadUint32((*uint32)(&C.threadExited)) != 0 {
			println("OK")
			return
		}
	}
}
