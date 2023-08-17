// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// #cgo noescape annotations for a C function means its arguments won't escape to heap.

/*
#cgo noescape runCWithNoEscape

void runCWithNoEscape(void *p) {
}
void runCWithoutNoEscape(void *p) {
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"unsafe"
)

var NUM uint64 = 100

func init() {
	register("CgoNoEscape", CgoNoEscape)
}

//go:noinline
func withNoEscape() {
	var str string
	C.runCWithNoEscape(unsafe.Pointer(&str))
}

//go:noinline
func withoutNoEscape() {
	var str string
	C.runCWithoutNoEscape(unsafe.Pointer(&str))
}

func CgoNoEscape() {
	// make GC nearly stop
	debug.SetGCPercent(1000)

	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	preHeapObjects := stats.HeapObjects

	for i := uint64(0); i < NUM; i++ {
		withNoEscape()
	}

	runtime.ReadMemStats(&stats)
	nowHeapObjects := stats.HeapObjects

	if nowHeapObjects-preHeapObjects >= NUM {
		fmt.Printf("too much heap objects allocated, pre: %v, now: %v\n", preHeapObjects, nowHeapObjects)
	}

	runtime.ReadMemStats(&stats)
	preHeapObjects = stats.HeapObjects

	for i := uint64(0); i < NUM; i++ {
		withoutNoEscape()
	}

	runtime.ReadMemStats(&stats)
	nowHeapObjects = stats.HeapObjects

	if nowHeapObjects-preHeapObjects < NUM {
		fmt.Printf("too less too heap objects allocated, pre: %v, now: %v\n", preHeapObjects, nowHeapObjects)
	}

	fmt.Println("OK")
}
