// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test taking a goroutine profile with C traceback.

/*
// Defined in gprof_c.c.
void CallGoSleep(void);
void gprofCgoTraceback(void* parg);
void gprofCgoContext(void* parg);
*/
import "C"

import (
	"fmt"
	"io"
	"runtime"
	"runtime/pprof"
	"time"
	"unsafe"
)

func init() {
	register("GoroutineProfile", GoroutineProfile)
}

func GoroutineProfile() {
	runtime.SetCgoTraceback(0, unsafe.Pointer(C.gprofCgoTraceback), unsafe.Pointer(C.gprofCgoContext), nil)

	go C.CallGoSleep()
	go C.CallGoSleep()
	go C.CallGoSleep()
	time.Sleep(1 * time.Second)

	prof := pprof.Lookup("goroutine")
	prof.WriteTo(io.Discard, 1)
	fmt.Println("OK")
}

//export GoSleep
func GoSleep() {
	time.Sleep(time.Hour)
}
