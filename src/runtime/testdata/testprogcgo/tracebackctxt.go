// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The __attribute__((weak)) used below doesn't seem to work on Windows.

// +build !windows

package main

// Test the context argument to SetCgoTraceback.
// Use fake context, traceback, and symbolizer functions.

/*
#include <stdlib.h>
#include <stdint.h>

// Use weak declarations so that we can define functions here even
// though we use //export in the Go code.
extern void tcContext(void*) __attribute__((weak));
extern void tcTraceback(void*) __attribute__((weak));
extern void tcSymbolizer(void*) __attribute__((weak));

extern void G1(void);
extern void G2(void);

static void C1() {
	G1();
}

static void C2() {
	G2();
}

struct cgoContextArg {
	uintptr_t context;
};

struct cgoTracebackArg {
	uintptr_t  context;
	uintptr_t* buf;
	uintptr_t  max;
};

struct cgoSymbolizerArg {
	uintptr_t   pc;
	const char* file;
	uintptr_t   lineno;
	const char* func;
	uintptr_t   entry;
	uintptr_t   more;
	uintptr_t   data;
};

// Global so that there is only one, weak so that //export works.
// Uses atomic adds and subtracts to catch the possibility of
// erroneous calls from multiple threads; that should be impossible in
// this test case, but we check just in case.
int contextCount __attribute__((weak));

static int getContextCount() {
	return __sync_add_and_fetch(&contextCount, 0);
}

void tcContext(void* parg) {
	struct cgoContextArg* arg = (struct cgoContextArg*)(parg);
	if (arg->context == 0) {
		arg->context = __sync_add_and_fetch(&contextCount, 1);
	} else {
		if (arg->context != __sync_add_and_fetch(&contextCount, 0)) {
			abort();
		}
		__sync_sub_and_fetch(&contextCount, 1);
	}
}

void tcTraceback(void* parg) {
	int base, i;
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
	if (arg->context == 0) {
		// This shouldn't happen in this program.
		abort();
	}
	// Return a variable number of PC values.
	base = arg->context << 8;
	for (i = 0; i < arg->context; i++) {
		if (i < arg->max) {
			arg->buf[i] = base + i;
		}
	}
}

void tcSymbolizer(void *parg) {
	struct cgoSymbolizerArg* arg = (struct cgoSymbolizerArg*)(parg);
	if (arg->pc == 0) {
		return;
	}
	// Report two lines per PC returned by traceback, to test more handling.
	arg->more = arg->file == NULL;
	arg->file = "tracebackctxt.go";
	arg->func = "cFunction";
	arg->lineno = arg->pc + (arg->more << 16);
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

func init() {
	register("TracebackContext", TracebackContext)
}

var tracebackOK bool

func TracebackContext() {
	runtime.SetCgoTraceback(0, unsafe.Pointer(C.tcTraceback), unsafe.Pointer(C.tcContext), unsafe.Pointer(C.tcSymbolizer))
	C.C1()
	if got := C.getContextCount(); got != 0 {
		fmt.Printf("at end contextCount == %d, expected 0\n", got)
		tracebackOK = false
	}
	if tracebackOK {
		fmt.Println("OK")
	}
}

//export G1
func G1() {
	C.C2()
}

//export G2
func G2() {
	pc := make([]uintptr, 32)
	n := runtime.Callers(0, pc)
	cf := runtime.CallersFrames(pc[:n])
	var frames []runtime.Frame
	for {
		frame, more := cf.Next()
		frames = append(frames, frame)
		if !more {
			break
		}
	}

	want := []struct {
		function string
		line     int
	}{
		{"main.G2", 0},
		{"cFunction", 0x10200},
		{"cFunction", 0x200},
		{"cFunction", 0x10201},
		{"cFunction", 0x201},
		{"main.G1", 0},
		{"cFunction", 0x10100},
		{"cFunction", 0x100},
		{"main.TracebackContext", 0},
	}

	ok := true
	i := 0
wantLoop:
	for _, w := range want {
		for ; i < len(frames); i++ {
			if w.function == frames[i].Function {
				if w.line != 0 && w.line != frames[i].Line {
					fmt.Printf("found function %s at wrong line %#x (expected %#x)\n", w.function, frames[i].Line, w.line)
					ok = false
				}
				i++
				continue wantLoop
			}
		}
		fmt.Printf("did not find function %s in\n", w.function)
		for _, f := range frames {
			fmt.Println(f)
		}
		ok = false
		break
	}
	tracebackOK = ok
	if got := C.getContextCount(); got != 2 {
		fmt.Printf("at bottom contextCount == %d, expected 2\n", got)
		tracebackOK = false
	}
}
