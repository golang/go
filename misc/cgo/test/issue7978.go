// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7978.  Stack tracing didn't work during cgo code after calling a Go
// callback.  Make sure GC works and the stack trace is correct.

package cgotest

/*
#include <stdint.h>

void issue7978cb(void);

#if defined(__APPLE__) && defined(__arm__)
// on Darwin/ARM, libSystem doesn't provide implementation of the __sync_fetch_and_add
// primitive, and although gcc supports it, it doesn't inline its definition.
// Clang could inline its definition, so we require clang on Darwin/ARM.
#if defined(__clang__)
#define HAS_SYNC_FETCH_AND_ADD 1
#else
#define HAS_SYNC_FETCH_AND_ADD 0
#endif
#else
#define HAS_SYNC_FETCH_AND_ADD 1
#endif

// use ugly atomic variable sync since that doesn't require calling back into
// Go code or OS dependencies
static void issue7978c(uint32_t *sync) {
#if HAS_SYNC_FETCH_AND_ADD
	while(__sync_fetch_and_add(sync, 0) != 0)
		;
	__sync_fetch_and_add(sync, 1);
	while(__sync_fetch_and_add(sync, 0) != 2)
		;
	issue7978cb();
	__sync_fetch_and_add(sync, 1);
	while(__sync_fetch_and_add(sync, 0) != 6)
		;
#endif
}
*/
import "C"

import (
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
)

var issue7978sync uint32

func issue7978check(t *testing.T, wantFunc string, badFunc string, depth int) {
	runtime.GC()
	buf := make([]byte, 65536)
	trace := string(buf[:runtime.Stack(buf, true)])
	for _, goroutine := range strings.Split(trace, "\n\n") {
		if strings.Contains(goroutine, "test.issue7978go") {
			trace := strings.Split(goroutine, "\n")
			// look for the expected function in the stack
			for i := 0; i < depth; i++ {
				if badFunc != "" && strings.Contains(trace[1+2*i], badFunc) {
					t.Errorf("bad stack: found %s in the stack:\n%s", badFunc, goroutine)
					return
				}
				if strings.Contains(trace[1+2*i], wantFunc) {
					return
				}
			}
			t.Errorf("bad stack: didn't find %s in the stack:\n%s", wantFunc, goroutine)
			return
		}
	}
	t.Errorf("bad stack: goroutine not found. Full stack dump:\n%s", trace)
}

func issue7978wait(store uint32, wait uint32) {
	if store != 0 {
		atomic.StoreUint32(&issue7978sync, store)
	}
	for atomic.LoadUint32(&issue7978sync) != wait {
		runtime.Gosched()
	}
}

//export issue7978cb
func issue7978cb() {
	issue7978wait(3, 4)
}

func issue7978go() {
	C.issue7978c((*C.uint32_t)(&issue7978sync))
	issue7978wait(7, 8)
}

func test7978(t *testing.T) {
	if runtime.Compiler == "gccgo" {
		t.Skip("gccgo can not do stack traces of C code")
	}
	if C.HAS_SYNC_FETCH_AND_ADD == 0 {
		t.Skip("clang required for __sync_fetch_and_add support on darwin/arm")
	}
	if os.Getenv("GOTRACEBACK") != "2" {
		t.Fatalf("GOTRACEBACK must be 2")
	}
	issue7978sync = 0
	go issue7978go()
	// test in c code, before callback
	issue7978wait(0, 1)
	issue7978check(t, "runtime.cgocall_errno(", "", 1)
	// test in go code, during callback
	issue7978wait(2, 3)
	issue7978check(t, "test.issue7978cb(", "test.issue7978go", 3)
	// test in c code, after callback
	issue7978wait(4, 5)
	issue7978check(t, "runtime.cgocall_errno(", "runtime.cgocallback", 1)
	// test in go code, after return from cgo
	issue7978wait(6, 7)
	issue7978check(t, "test.issue7978go(", "", 3)
	atomic.StoreUint32(&issue7978sync, 8)
}
