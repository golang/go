// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure unsafe-uintptr arguments are not
// kept alive longer than expected.

package main

import (
	"runtime"
	"sync/atomic"
	"unsafe"
)

var done uint32

func setup() unsafe.Pointer {
	s := "ok"
	runtime.SetFinalizer(&s, func(p *string) { atomic.StoreUint32(&done, 1) })
	return unsafe.Pointer(&s)
}

//go:noinline
//go:uintptrescapes
func before(p uintptr) int {
	runtime.GC()
	if atomic.LoadUint32(&done) != 0 {
		panic("GC early")
	}
	return 0
}

func after() int {
	runtime.GC()
	if atomic.LoadUint32(&done) == 0 {
		panic("GC late")
	}
	return 0
}

func main() {
	_ = before(uintptr(setup())) + after()
}
