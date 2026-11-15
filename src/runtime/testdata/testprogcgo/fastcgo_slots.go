// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package main

/*
#cgo noescape slots_wait
#cgo nocallback slots_wait

#include <stdlib.h>

int slots_wait(unsigned char *p, unsigned char want, volatile int *entered, volatile int *release) {
	volatile unsigned char *vp = (volatile unsigned char *)p;
	*entered = 1;
	for (unsigned long long i = 0; i < 4000000000ULL; i++) {
		if (*vp != want) {
			return 1;
		}
		if (*release != 0) {
			return 0;
		}
	}
	return 2;
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"
)

func init() {
	register("LogosFastCgoSlotsNotRoots", LogosFastCgoSlotsNotRoots)
}

func slotsGCs() uint32 {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ms.NumGC
}

func LogosFastCgoSlotsNotRoots() {
	done := make(chan struct{}, 1)
	res := make(chan int32, 1)
	entered := (*int32)(C.calloc(1, C.size_t(unsafe.Sizeof(C.int(0)))))
	release := (*int32)(C.calloc(1, C.size_t(unsafe.Sizeof(C.int(0)))))
	defer C.free(unsafe.Pointer(entered))
	defer C.free(unsafe.Pointer(release))
	if entered == nil || release == nil {
		fmt.Println("FAIL: out of memory")
		return
	}
	go func() {
		b := new([256 << 10]byte)
		b[0] = 42
		runtime.AddCleanup(b, func(ch chan struct{}) { ch <- struct{}{} }, done)
		res <- int32(C.slots_wait((*C.uchar)(unsafe.Pointer(&b[0])), C.uchar(42), (*C.int)(unsafe.Pointer(entered)), (*C.int)(unsafe.Pointer(release))))
	}()
	for atomic.LoadInt32(entered) == 0 {
		time.Sleep(time.Millisecond)
	}
	before := slotsGCs()
	runtime.GC()
	during := slotsGCs() - before
	if during < 1 {
		atomic.StoreInt32(release, 1)
		<-res
		fmt.Println("FAIL: no GC cycle completed during the fast call")
		return
	}
	atomic.StoreInt32(release, 1)
	var code int32
	select {
	case code = <-res:
	case <-time.After(60 * time.Second):
		fmt.Println("FAIL: fast call did not return within 60s")
		return
	}
	if code != 0 {
		fmt.Println("FAIL: fast-call argument changed during collection")
		return
	}
	deadline := time.Now().Add(30 * time.Second)
	for {
		select {
		case <-done:
			fmt.Println("OK")
			return
		default:
		}
		if time.Now().After(deadline) {
			fmt.Println("FAIL: object referenced only by fast-call state survived collection")
			return
		}
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
}
