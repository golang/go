// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The __attribute__((weak)) used below doesn't seem to work on Windows.

package main

// Test the context argument to SetCgoTraceback.
// Use fake context, traceback, and symbolizer functions.

/*
// Defined in tracebackctxt_c.c.
extern void C1(void);
extern void C2(void);
extern void tcContext(void*);
extern void tcTraceback(void*);
extern void tcSymbolizer(void*);
extern int getContextCount(void);
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
