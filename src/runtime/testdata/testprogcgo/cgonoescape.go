// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// #cgo noescape annotations for a C function means its arguments won't escape to heap.

// We assume that there won't be 100 new allocated heap objects in other places,
// i.e. runtime.ReadMemStats or other runtime background works.
// So, the tests are:
// 1. at least 100 new allocated heap objects after invoking withoutNoEscape 100 times.
// 2. less than 100 new allocated heap objects after invoking withoutNoEscape 100 times.

/*
#cgo noescape runCWithNoEscape
#cgo nocallback runCWithNoEscape

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

const num = 100

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
	// make GC stop to see the heap objects allocated
	debug.SetGCPercent(-1)

	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	preHeapObjects := stats.HeapObjects

	for i := 0; i < num; i++ {
		withNoEscape()
	}

	runtime.ReadMemStats(&stats)
	nowHeapObjects := stats.HeapObjects

	if nowHeapObjects-preHeapObjects >= num {
		fmt.Printf("too many heap objects allocated, pre: %v, now: %v\n", preHeapObjects, nowHeapObjects)
	}

	runtime.ReadMemStats(&stats)
	preHeapObjects = stats.HeapObjects

	for i := 0; i < num; i++ {
		withoutNoEscape()
	}

	runtime.ReadMemStats(&stats)
	nowHeapObjects = stats.HeapObjects

	if nowHeapObjects-preHeapObjects < num {
		fmt.Printf("too few heap objects allocated, pre: %v, now: %v\n", preHeapObjects, nowHeapObjects)
	}

	fmt.Println("OK")
}
