// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"time"
	"unsafe"
)

func init() {
	register("CheckPtrAlignmentNoPtr", CheckPtrAlignmentNoPtr)
	register("CheckPtrAlignmentPtr", CheckPtrAlignmentPtr)
	register("CheckPtrAlignmentNilPtr", CheckPtrAlignmentNilPtr)
	register("CheckPtrArithmetic", CheckPtrArithmetic)
	register("CheckPtrArithmetic2", CheckPtrArithmetic2)
	register("CheckPtrSize", CheckPtrSize)
	register("CheckPtrSmall", CheckPtrSmall)
	register("CheckPtrSliceOK", CheckPtrSliceOK)
	register("CheckPtrSliceFail", CheckPtrSliceFail)
}

func CheckPtrAlignmentNoPtr() {
	var x [2]int64
	p := unsafe.Pointer(&x[0])
	sink2 = (*int64)(unsafe.Pointer(uintptr(p) + 1))
}

func CheckPtrAlignmentPtr() {
	var x [2]int64
	p := unsafe.Pointer(&x[0])
	sink2 = (**int64)(unsafe.Pointer(uintptr(p) + 1))
}

// CheckPtrAlignmentNilPtr tests that checkptrAlignment doesn't crash
// on nil pointers (#47430).
func CheckPtrAlignmentNilPtr() {
	var do func(int)
	do = func(n int) {
		// Inflate the stack so runtime.shrinkstack gets called during GC
		if n > 0 {
			do(n - 1)
		}

		var p unsafe.Pointer
		_ = (*int)(p)
	}

	go func() {
		for {
			runtime.GC()
		}
	}()

	go func() {
		for i := 0; ; i++ {
			do(i % 1024)
		}
	}()

	time.Sleep(time.Second)
}

func CheckPtrArithmetic() {
	var x int
	i := uintptr(unsafe.Pointer(&x))
	sink2 = (*int)(unsafe.Pointer(i))
}

func CheckPtrArithmetic2() {
	var x [2]int64
	p := unsafe.Pointer(&x[1])
	var one uintptr = 1
	sink2 = unsafe.Pointer(uintptr(p) & ^one)
}

func CheckPtrSize() {
	p := new(int64)
	sink2 = p
	sink2 = (*[100]int64)(unsafe.Pointer(p))
}

func CheckPtrSmall() {
	sink2 = unsafe.Pointer(uintptr(1))
}

func CheckPtrSliceOK() {
	p := new([4]int64)
	sink2 = unsafe.Slice(&p[1], 3)
}

func CheckPtrSliceFail() {
	p := new(int64)
	sink2 = p
	sink2 = unsafe.Slice(p, 100)
}
