// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program will crash.
// We want the stack trace to include the C functions.
// We use a fake traceback, and a symbolizer that dumps a string we recognize.

/*
#cgo CFLAGS: -g -O0

// Defined in traceback_c.c.
extern int crashInGo;
int tracebackF1(void);
void cgoTraceback(void* parg);
void cgoSymbolizer(void* parg);
*/
import "C"

import (
	"runtime"
	"unsafe"
)

func init() {
	register("CrashTraceback", CrashTraceback)
	register("CrashTracebackGo", CrashTracebackGo)
}

func CrashTraceback() {
	runtime.SetCgoTraceback(0, unsafe.Pointer(C.cgoTraceback), nil, unsafe.Pointer(C.cgoSymbolizer))
	C.tracebackF1()
}

func CrashTracebackGo() {
	C.crashInGo = 1
	CrashTraceback()
}

//export h1
func h1() {
	h2()
}

func h2() {
	h3()
}

func h3() {
	var x *int
	*x = 0
}
