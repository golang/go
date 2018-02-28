// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 17381: make sure leave function with non-empty frame
// saves link register, so that traceback will work.

package main

import (
	"runtime"
	"unsafe"
)

func main() {
	defer func() {
		if recover() == nil {
			panic("did not panic")
		}
		pcs := make([]uintptr, 20)
		n := runtime.Callers(1, pcs)
		for _, pc := range pcs[:n] {
			if runtime.FuncForPC(pc).Name() == "main.main" {
				return
			}
		}
		panic("cannot find main.main in backtrace")
	}()

	prep()
	f() // should panic
}

func funcPC(f interface{}) uintptr {
	var ptr uintptr
	return **(**uintptr)(unsafe.Pointer(uintptr(unsafe.Pointer(&f)) + unsafe.Sizeof(ptr)))
}

//go:noinline
func f() {
	var t [1]int // non-empty frame
	*(*int)(nil) = t[0]
}

var p = funcPC(runtime.GC) + 8

//go:noinline
func prep() {
	// put some garbage on stack
	var x = [20]uintptr{p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p}
	_ = x
}
