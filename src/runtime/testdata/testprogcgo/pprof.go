// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Run a slow C function saving a CPU profile.

/*
#include <stdint.h>

int salt1;
int salt2;

void cpuHog() {
	int foo = salt1;
	int i;

	for (i = 0; i < 100000; i++) {
		if (foo > 0) {
			foo *= foo;
		} else {
			foo *= foo + 1;
		}
	}
	salt2 = foo;
}

void cpuHog2() {
}

static int cpuHogCount;

struct cgoTracebackArg {
	uintptr_t  context;
	uintptr_t  sigContext;
	uintptr_t* buf;
	uintptr_t  max;
};

// pprofCgoTraceback is passed to runtime.SetCgoTraceback.
// For testing purposes it pretends that all CPU hits in C code are in cpuHog.
// Issue #29034: At least 2 frames are required to verify all frames are captured
// since runtime/pprof ignores the runtime.goexit base frame if it exists.
void pprofCgoTraceback(void* parg) {
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
	arg->buf[0] = (uintptr_t)(cpuHog) + 0x10;
	arg->buf[1] = (uintptr_t)(cpuHog2) + 0x4;
	arg->buf[2] = 0;
	++cpuHogCount;
}

// getCpuHogCount fetches the number of times we've seen cpuHog in the
// traceback.
int getCpuHogCount() {
	return cpuHogCount;
}
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
	"unsafe"
)

func init() {
	register("CgoPprof", CgoPprof)
}

func CgoPprof() {
	runtime.SetCgoTraceback(0, unsafe.Pointer(C.pprofCgoTraceback), nil, nil)

	f, err := os.CreateTemp("", "prof")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	if err := pprof.StartCPUProfile(f); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	t0 := time.Now()
	for C.getCpuHogCount() < 2 && time.Since(t0) < time.Second {
		C.cpuHog()
	}

	pprof.StopCPUProfile()

	name := f.Name()
	if err := f.Close(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	fmt.Println(name)
}
