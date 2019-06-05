// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program outputs a CPU profile that includes
// both Go and Cgo stacks. This is used by the mapping info
// tests in runtime/pprof.
//
// If SETCGOTRACEBACK=1 is set, the CPU profile will includes
// PCs from C side but they will not be symbolized.
package main

/*
#include <stdint.h>
#include <stdlib.h>

int cpuHogCSalt1 = 0;
int cpuHogCSalt2 = 0;

void CPUHogCFunction() {
	int foo = cpuHogCSalt1;
	int i;
	for (i = 0; i < 100000; i++) {
		if (foo > 0) {
			foo *= foo;
		} else {
			foo *= foo + 1;
		}
		cpuHogCSalt2 = foo;
	}
}

struct CgoTracebackArg {
	uintptr_t context;
        uintptr_t sigContext;
	uintptr_t *buf;
        uintptr_t max;
};

void CollectCgoTraceback(void* parg) {
        struct CgoTracebackArg* arg = (struct CgoTracebackArg*)(parg);
	arg->buf[0] = (uintptr_t)(CPUHogCFunction);
	arg->buf[1] = 0;
};
*/
import "C"

import (
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
	"unsafe"
)

func init() {
	if v := os.Getenv("SETCGOTRACEBACK"); v == "1" {
		// Collect some PCs from C-side, but don't symbolize.
		runtime.SetCgoTraceback(0, unsafe.Pointer(C.CollectCgoTraceback), nil, nil)
	}
}

func main() {
	go cpuHogGoFunction()
	go cpuHogCFunction()
	runtime.Gosched()

	if err := pprof.StartCPUProfile(os.Stdout); err != nil {
		log.Fatal("can't start CPU profile: ", err)
	}
	time.Sleep(200 * time.Millisecond)
	pprof.StopCPUProfile()

	if err := os.Stdout.Close(); err != nil {
		log.Fatal("can't write CPU profile: ", err)
	}
}

var salt1 int
var salt2 int

func cpuHogGoFunction() {
	// Generates CPU profile samples including a Go call path.
	for {
		foo := salt1
		for i := 0; i < 1e5; i++ {
			if foo > 0 {
				foo *= foo
			} else {
				foo *= foo + 1
			}
			salt2 = foo
		}
		runtime.Gosched()
	}
}

func cpuHogCFunction() {
	// Generates CPU profile samples including a Cgo call path.
	for {
		C.CPUHogCFunction()
		runtime.Gosched()
	}
}
