// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program will crash.
// We want the stack trace to include the C functions.
// We use a fake traceback, and a symbolizer that dumps a string we recognize.

/*
#cgo CFLAGS: -g -O0

#include <stdint.h>

char *p;

static int f3(void) {
	*p = 0;
	return 0;
}

static int f2(void) {
	return f3();
}

static int f1(void) {
	return f2();
}

struct cgoTracebackArg {
	uintptr_t  context;
	uintptr_t  sigContext;
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

void cgoTraceback(void* parg) {
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
	arg->buf[0] = 1;
	arg->buf[1] = 2;
	arg->buf[2] = 3;
	arg->buf[3] = 0;
}

void cgoSymbolizer(void* parg) {
	struct cgoSymbolizerArg* arg = (struct cgoSymbolizerArg*)(parg);
	if (arg->pc != arg->data + 1) {
		arg->file = "unexpected data";
	} else {
		arg->file = "cgo symbolizer";
	}
	arg->lineno = arg->data + 1;
	arg->data++;
}
*/
import "C"

import (
	"runtime"
	"unsafe"
)

func init() {
	register("CrashTraceback", CrashTraceback)
}

func CrashTraceback() {
	runtime.SetCgoTraceback(0, unsafe.Pointer(C.cgoTraceback), nil, unsafe.Pointer(C.cgoSymbolizer))
	C.f1()
}
