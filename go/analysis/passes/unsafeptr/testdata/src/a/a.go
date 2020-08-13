// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"reflect"
	"unsafe"
)

func f() {
	var x unsafe.Pointer
	var y uintptr
	x = unsafe.Pointer(y) // want "possible misuse of unsafe.Pointer"
	y = uintptr(x)

	// only allowed pointer arithmetic is ptr +/-/&^ num.
	// num+ptr is technically okay but still flagged: write ptr+num instead.
	x = unsafe.Pointer(uintptr(x) + 1)
	x = unsafe.Pointer(((uintptr((x))) + 1))
	x = unsafe.Pointer(1 + uintptr(x))          // want "possible misuse of unsafe.Pointer"
	x = unsafe.Pointer(uintptr(x) + uintptr(x)) // want "possible misuse of unsafe.Pointer"
	x = unsafe.Pointer(uintptr(x) - 1)
	x = unsafe.Pointer(1 - uintptr(x)) // want "possible misuse of unsafe.Pointer"
	x = unsafe.Pointer(uintptr(x) &^ 3)
	x = unsafe.Pointer(1 &^ uintptr(x)) // want "possible misuse of unsafe.Pointer"

	// certain uses of reflect are okay
	var v reflect.Value
	x = unsafe.Pointer(v.Pointer())
	x = unsafe.Pointer(v.Pointer() + 1) // want "possible misuse of unsafe.Pointer"
	x = unsafe.Pointer(v.UnsafeAddr())
	x = unsafe.Pointer((v.UnsafeAddr()))
	var s1 *reflect.StringHeader
	x = unsafe.Pointer(s1.Data)
	x = unsafe.Pointer(s1.Data + 1) // want "possible misuse of unsafe.Pointer"
	var s2 *reflect.SliceHeader
	x = unsafe.Pointer(s2.Data)
	var s3 reflect.StringHeader
	x = unsafe.Pointer(s3.Data) // want "possible misuse of unsafe.Pointer"
	var s4 reflect.SliceHeader
	x = unsafe.Pointer(s4.Data) // want "possible misuse of unsafe.Pointer"

	// but only in reflect
	var vv V
	x = unsafe.Pointer(vv.Pointer())    // want "possible misuse of unsafe.Pointer"
	x = unsafe.Pointer(vv.UnsafeAddr()) // want "possible misuse of unsafe.Pointer"
	var ss1 *StringHeader
	x = unsafe.Pointer(ss1.Data) // want "possible misuse of unsafe.Pointer"
	var ss2 *SliceHeader
	x = unsafe.Pointer(ss2.Data) // want "possible misuse of unsafe.Pointer"

}

type V interface {
	Pointer() uintptr
	UnsafeAddr() uintptr
}

type StringHeader struct {
	Data uintptr
}

type SliceHeader struct {
	Data uintptr
}
