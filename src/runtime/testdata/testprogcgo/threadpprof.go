// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

package main

// Run a slow C function saving a CPU profile.

/*
#include <stdint.h>
#include <time.h>
#include <pthread.h>

int threadSalt1;
int threadSalt2;

static pthread_t tid;

void cpuHogThread() {
	int foo = threadSalt1;
	int i;

	for (i = 0; i < 100000; i++) {
		if (foo > 0) {
			foo *= foo;
		} else {
			foo *= foo + 1;
		}
	}
	threadSalt2 = foo;
}

void cpuHogThread2() {
}

struct cgoTracebackArg {
	uintptr_t  context;
	uintptr_t  sigContext;
	uintptr_t* buf;
	uintptr_t  max;
};

// pprofCgoThreadTraceback is passed to runtime.SetCgoTraceback.
// For testing purposes it pretends that all CPU hits on the cpuHog
// C thread are in cpuHog.
void pprofCgoThreadTraceback(void* parg) {
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
	if (pthread_self() == tid) {
		arg->buf[0] = (uintptr_t)(cpuHogThread) + 0x10;
		arg->buf[1] = (uintptr_t)(cpuHogThread2) + 0x4;
		arg->buf[2] = 0;
	} else
		arg->buf[0] = 0;
}

static void* cpuHogDriver(void* arg __attribute__ ((unused))) {
	while (1) {
		cpuHogThread();
	}
	return 0;
}

void runCPUHogThread(void) {
	pthread_create(&tid, 0, cpuHogDriver, 0);
}
*/
import "C"

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
	"unsafe"
)

func init() {
	register("CgoPprofThread", CgoPprofThread)
	register("CgoPprofThreadNoTraceback", CgoPprofThreadNoTraceback)
}

func CgoPprofThread() {
	runtime.SetCgoTraceback(0, unsafe.Pointer(C.pprofCgoThreadTraceback), nil, nil)
	pprofThread()
}

func CgoPprofThreadNoTraceback() {
	pprofThread()
}

func pprofThread() {
	f, err := os.CreateTemp("", "prof")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	if err := pprof.StartCPUProfile(f); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	// This goroutine may receive a profiling signal while creating the C-owned
	// thread. If it does, the SetCgoTraceback handler will make the leaf end of
	// the stack look almost (but not exactly) like the stacks the test case is
	// trying to find. Attach a profiler label so the test can filter out those
	// confusing samples.
	pprof.Do(context.Background(), pprof.Labels("ignore", "ignore"), func(ctx context.Context) {
		C.runCPUHogThread()
	})

	time.Sleep(1 * time.Second)

	pprof.StopCPUProfile()

	name := f.Name()
	if err := f.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	fmt.Println(name)
}
