// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

package main

/*
#include <stdint.h>
#include <signal.h>
#include <pthread.h>

volatile int32_t spinlock;

static void *thread1(void *p) {
	(void)p;
	while (spinlock == 0)
		;
	pthread_kill(pthread_self(), SIGPROF);
	spinlock = 0;
	return NULL;
}
__attribute__((constructor)) void issue9456() {
	pthread_t tid;
	pthread_create(&tid, 0, thread1, NULL);
}

void **nullptr;

void *crash(void *p) {
	*nullptr = p;
	return 0;
}

int start_crashing_thread(void) {
	pthread_t tid;
	return pthread_create(&tid, 0, crash, 0);
}
*/
import "C"

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"
)

func init() {
	register("CgoExternalThreadSIGPROF", CgoExternalThreadSIGPROF)
	register("CgoExternalThreadSignal", CgoExternalThreadSignal)
}

func CgoExternalThreadSIGPROF() {
	// This test intends to test that sending SIGPROF to foreign threads
	// before we make any cgo call will not abort the whole process, so
	// we cannot make any cgo call here. See https://golang.org/issue/9456.
	atomic.StoreInt32((*int32)(unsafe.Pointer(&C.spinlock)), 1)
	for atomic.LoadInt32((*int32)(unsafe.Pointer(&C.spinlock))) == 1 {
		runtime.Gosched()
	}
	println("OK")
}

func CgoExternalThreadSignal() {
	if len(os.Args) > 2 && os.Args[2] == "crash" {
		i := C.start_crashing_thread()
		if i != 0 {
			fmt.Println("pthread_create failed:", i)
			// Exit with 0 because parent expects us to crash.
			return
		}

		// We should crash immediately, but give it plenty of
		// time before failing (by exiting 0) in case we are
		// running on a slow system.
		time.Sleep(5 * time.Second)
		return
	}

	out, err := exec.Command(os.Args[0], "CgoExternalThreadSignal", "crash").CombinedOutput()
	if err == nil {
		fmt.Println("C signal did not crash as expected\n")
		fmt.Printf("%s\n", out)
		os.Exit(1)
	}

	fmt.Println("OK")
}
