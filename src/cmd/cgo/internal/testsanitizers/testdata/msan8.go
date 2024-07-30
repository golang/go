// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <pthread.h>
#include <signal.h>
#include <stdint.h>

#include <sanitizer/msan_interface.h>

// cgoTracebackArg is the type of the argument passed to msanGoTraceback.
struct cgoTracebackArg {
	uintptr_t context;
	uintptr_t sigContext;
	uintptr_t* buf;
	uintptr_t max;
};

// msanGoTraceback is registered as the cgo traceback function.
// This will be called when a signal occurs.
void msanGoTraceback(void* parg) {
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
        arg->buf[0] = 0;
}

// Don't warn if the compiler doesn't support the maybe_undef attribute.
#pragma GCC diagnostic ignored "-Wattributes"

// msanGoWait will be called with all registers undefined as far as
// msan is concerned. It just waits for a signal.
// Because the registers are msan-undefined, the signal handler will
// be invoked with all registers msan-undefined.
// The maybe_undef attribute tells clang to not complain about
// passing uninitialized values.
__attribute__((noinline))
void msanGoWait(unsigned long a1 __attribute__((maybe_undef)),
		unsigned long a2 __attribute__((maybe_undef)),
		unsigned long a3 __attribute__((maybe_undef)),
		unsigned long a4 __attribute__((maybe_undef)),
		unsigned long a5 __attribute__((maybe_undef)),
		unsigned long a6 __attribute__((maybe_undef))) {
	sigset_t mask;

	sigemptyset(&mask);
        sigsuspend(&mask);
}

// msanGoSignalThread is the thread ID of the msanGoLoop thread.
static pthread_t msanGoSignalThread;

// msanGoSignalThreadSet is used to record that msanGoSignalThread
// has been initialized. This is accessed atomically.
static int32_t msanGoSignalThreadSet;

// uninit is explicitly poisoned, so that we can make all registers
// undefined by calling msanGoWait.
static unsigned long uninit;

// msanGoLoop loops calling msanGoWait, with the arguments passed
// such that msan thinks that they are undefined. msan permits
// undefined values to be used as long as they are not used to
// for conditionals or for memory access.
void msanGoLoop() {
	int i;

	msanGoSignalThread = pthread_self();
        __atomic_store_n(&msanGoSignalThreadSet, 1, __ATOMIC_SEQ_CST);

	// Force uninit to be undefined for msan.
	__msan_poison(&uninit, sizeof uninit);
	for (i = 0; i < 100; i++) {
		msanGoWait(uninit, uninit, uninit, uninit, uninit, uninit);
        }
}

// msanGoReady returns whether msanGoSignalThread is set.
int msanGoReady() {
	return __atomic_load_n(&msanGoSignalThreadSet, __ATOMIC_SEQ_CST) != 0;
}

// msanGoSendSignal sends a signal to the msanGoLoop thread.
void msanGoSendSignal() {
	pthread_kill(msanGoSignalThread, SIGWINCH);
}
*/
import "C"

import (
	"runtime"
	"time"
)

func main() {
	runtime.SetCgoTraceback(0, C.msanGoTraceback, nil, nil)

	c := make(chan bool)
	go func() {
		defer func() { c <- true }()
		C.msanGoLoop()
	}()

	for C.msanGoReady() == 0 {
		time.Sleep(time.Microsecond)
	}

loop:
	for {
		select {
		case <-c:
			break loop
		default:
			C.msanGoSendSignal()
			time.Sleep(time.Microsecond)
		}
	}
}
