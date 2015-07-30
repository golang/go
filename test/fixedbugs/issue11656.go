// run

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// darwin/386 seems to mangle the PC and SP before
// it manages to invoke the signal handler, so this test fails there.
// +build !darwin !386
//
// openbsd/386 and netbsd/386 don't work, not sure why.
// +build !openbsd !386
// +build !netbsd !386
//
// windows doesn't work, because Windows exception handling
// delivers signals based on the current PC, and that current PC
// doesn't go into the Go runtime.
// +build !windows
//
// arm64 gets "illegal instruction" (why is the data executable?)
// and is unable to do the traceback correctly (why?).
// +build !arm64

package main

import (
	"runtime"
	"runtime/debug"
	"unsafe"
)

func main() {
	debug.SetPanicOnFault(true)
	defer func() {
		if err := recover(); err == nil {
			panic("not panicking")
		}
		pc, _, _, _ := runtime.Caller(10)
		f := runtime.FuncForPC(pc)
		if f == nil || f.Name() != "main.f" {
			if f == nil {
				println("no func for ", unsafe.Pointer(pc))
			} else {
				println("found func:", f.Name())
			}
			panic("cannot find main.f on stack")
		}
	}()
	f(20)
}

func f(n int) {
	if n > 0 {
		f(n - 1)
	}
	var f struct {
		x uintptr
	}
	f.x = uintptr(unsafe.Pointer(&f))
	fn := *(*func())(unsafe.Pointer(&f))
	fn()
}
