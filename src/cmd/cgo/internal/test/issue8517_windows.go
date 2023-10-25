// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

//void testHandleLeaks();
import "C"

import (
	"syscall"
	"testing"
	"unsafe"
)

var issue8517counter int

var (
	kernel32              = syscall.MustLoadDLL("kernel32.dll")
	getProcessHandleCount = kernel32.MustFindProc("GetProcessHandleCount")
)

func processHandleCount(t *testing.T) int {
	const current_process = ^uintptr(0)
	var c uint32
	r, _, err := getProcessHandleCount.Call(current_process, uintptr(unsafe.Pointer(&c)))
	if r == 0 {
		t.Fatal(err)
	}
	return int(c)
}

func test8517(t *testing.T) {
	c1 := processHandleCount(t)
	C.testHandleLeaks()
	c2 := processHandleCount(t)
	if c1+issue8517counter <= c2 {
		t.Fatalf("too many handles leaked: issue8517counter=%v c1=%v c2=%v", issue8517counter, c1, c2)
	}
}

//export testHandleLeaksCallback
func testHandleLeaksCallback() {
	issue8517counter++
}
