// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func init() {
	register("CheckPtrAlignmentNoPtr", CheckPtrAlignmentNoPtr)
	register("CheckPtrAlignmentPtr", CheckPtrAlignmentPtr)
	register("CheckPtrArithmetic", CheckPtrArithmetic)
	register("CheckPtrSize", CheckPtrSize)
	register("CheckPtrSmall", CheckPtrSmall)
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

func CheckPtrArithmetic() {
	var x int
	i := uintptr(unsafe.Pointer(&x))
	sink2 = (*int)(unsafe.Pointer(i))
}

func CheckPtrSize() {
	p := new(int64)
	sink2 = p
	sink2 = (*[100]int64)(unsafe.Pointer(p))
}

func CheckPtrSmall() {
	sink2 = unsafe.Pointer(uintptr(1))
}
