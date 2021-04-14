// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that //go:uintptrescapes works for methods.

package main

import (
	"fmt"
	"runtime"
	"unsafe"
)

var callback func()

//go:noinline
//go:uintptrescapes
func F(ptr uintptr) { callback() }

//go:noinline
//go:uintptrescapes
func Fv(ptrs ...uintptr) { callback() }

type T struct{}

//go:noinline
//go:uintptrescapes
func (T) M(ptr uintptr) { callback() }

//go:noinline
//go:uintptrescapes
func (T) Mv(ptrs ...uintptr) { callback() }

// Each test should pass uintptr(ptr) as an argument to a function call,
// which in turn should call callback. The callback checks that ptr is kept alive.
var tests = []func(ptr unsafe.Pointer){
	func(ptr unsafe.Pointer) { F(uintptr(ptr)) },
	func(ptr unsafe.Pointer) { Fv(uintptr(ptr)) },
	func(ptr unsafe.Pointer) { T{}.M(uintptr(ptr)) },
	func(ptr unsafe.Pointer) { T{}.Mv(uintptr(ptr)) },
}

func main() {
	for i, test := range tests {
		finalized := false

		ptr := new([64]byte)
		runtime.SetFinalizer(ptr, func(*[64]byte) {
			finalized = true
		})

		callback = func() {
			runtime.GC()
			if finalized {
				fmt.Printf("test #%d failed\n", i)
			}
		}
		test(unsafe.Pointer(ptr))
	}
}
