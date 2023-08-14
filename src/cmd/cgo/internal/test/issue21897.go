// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && cgo && !internal

package cgotest

/*
#cgo LDFLAGS: -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
*/
import "C"
import (
	"runtime/debug"
	"testing"
	"unsafe"
)

func test21897(t *testing.T) {
	// Please write barrier, kick in soon.
	defer debug.SetGCPercent(debug.SetGCPercent(1))

	for i := 0; i < 10000; i++ {
		testCFNumberRef()
		testCFDateRef()
		testCFBooleanRef()
		// Allocate some memory, so eventually the write barrier is enabled
		// and it will see writes of bad pointers in the test* functions below.
		byteSliceSink = make([]byte, 1024)
	}
}

var byteSliceSink []byte

func testCFNumberRef() {
	var v int64 = 0
	xCFNumberRef = C.CFNumberCreate(C.kCFAllocatorSystemDefault, C.kCFNumberSInt64Type, unsafe.Pointer(&v))
	//fmt.Printf("CFNumberRef: %x\n", uintptr(unsafe.Pointer(xCFNumberRef)))
}

var xCFNumberRef C.CFNumberRef

func testCFDateRef() {
	xCFDateRef = C.CFDateCreate(C.kCFAllocatorSystemDefault, 0) // 0 value is 1 Jan 2001 00:00:00 GMT
	//fmt.Printf("CFDateRef: %x\n", uintptr(unsafe.Pointer(xCFDateRef)))
}

var xCFDateRef C.CFDateRef

func testCFBooleanRef() {
	xCFBooleanRef = C.kCFBooleanFalse
	//fmt.Printf("CFBooleanRef: %x\n", uintptr(unsafe.Pointer(xCFBooleanRef)))
}

var xCFBooleanRef C.CFBooleanRef
