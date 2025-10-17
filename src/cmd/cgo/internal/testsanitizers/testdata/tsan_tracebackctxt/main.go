// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
// Defined in tracebackctxt_c.c.
extern void C1(void);
extern void C2(void);
extern void tcContext(void*);
extern void tcTraceback(void*);
extern void tcSymbolizer(void*);
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Regression test for https://go.dev/issue/73949. TSAN should not report races
// on writes to the argument passed to the symbolizer function.
//
// Triggering this race requires calls to the symbolizer function with the same
// argument pointer on multiple threads. The runtime passes a stack variable to
// this function, so that means we need to get a single goroutine to execute on
// two threads, calling the symbolizer function on each.
//
// runtime.CallersFrames / Next will call the symbolizer function (if there are
// C frames). So the approach here is, with GOMAXPROCS=2, have 2 goroutines
// that use CallersFrames over and over, both frequently calling Gosched in an
// attempt to get picked up by the other P.

var tracebackOK bool

func main() {
	runtime.GOMAXPROCS(2)
	runtime.SetCgoTraceback(0, unsafe.Pointer(C.tcTraceback), unsafe.Pointer(C.tcContext), unsafe.Pointer(C.tcSymbolizer))
	C.C1()
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

	var wg sync.WaitGroup
	for range 2 {
		wg.Go(func() {
			for range 1000 {
				cf := runtime.CallersFrames(pc[:n])
				var frames []runtime.Frame
				for {
					frame, more := cf.Next()
					frames = append(frames, frame)
					if !more {
						break
					}
				}
				runtime.Gosched()
			}
		})
	}
	wg.Wait()

	tracebackOK = true
}
