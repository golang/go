// errorcheck -0 -m -l

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for unsafe.Pointer rules.

package escape

import (
	"reflect"
	"unsafe"
)

// (1) Conversion of a *T1 to Pointer to *T2.

func convert(p *float64) *uint64 { // ERROR "leaking param: p to result ~r1 level=0$"
	return (*uint64)(unsafe.Pointer(p))
}

// (3) Conversion of a Pointer to a uintptr and back, with arithmetic.

func arithAdd() unsafe.Pointer {
	var x [2]byte // ERROR "moved to heap: x"
	return unsafe.Pointer(uintptr(unsafe.Pointer(&x[0])) + 1)
}

func arithSub() unsafe.Pointer {
	var x [2]byte // ERROR "moved to heap: x"
	return unsafe.Pointer(uintptr(unsafe.Pointer(&x[1])) - 1)
}

func arithMask() unsafe.Pointer {
	var x [2]byte // ERROR "moved to heap: x"
	return unsafe.Pointer(uintptr(unsafe.Pointer(&x[1])) &^ 1)
}

// (5) Conversion of the result of reflect.Value.Pointer or
// reflect.Value.UnsafeAddr from uintptr to Pointer.

// BAD: should be "leaking param: p to result ~r1 level=0$"
func valuePointer(p *int) unsafe.Pointer { // ERROR "leaking param: p$"
	return unsafe.Pointer(reflect.ValueOf(p).Pointer())
}

// BAD: should be "leaking param: p to result ~r1 level=0$"
func valueUnsafeAddr(p *int) unsafe.Pointer { // ERROR "leaking param: p$"
	return unsafe.Pointer(reflect.ValueOf(p).Elem().UnsafeAddr())
}

// (6) Conversion of a reflect.SliceHeader or reflect.StringHeader
// Data field to or from Pointer.

func fromSliceData(s []int) unsafe.Pointer { // ERROR "leaking param: s to result ~r1 level=0$"
	return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&s)).Data)
}

func fromStringData(s string) unsafe.Pointer { // ERROR "leaking param: s to result ~r1 level=0$"
	return unsafe.Pointer((*reflect.StringHeader)(unsafe.Pointer(&s)).Data)
}

func toSliceData(s *[]int, p unsafe.Pointer) { // ERROR "s does not escape" "leaking param: p$"
	(*reflect.SliceHeader)(unsafe.Pointer(s)).Data = uintptr(p)
}

func toStringData(s *string, p unsafe.Pointer) { // ERROR "s does not escape" "leaking param: p$"
	(*reflect.SliceHeader)(unsafe.Pointer(s)).Data = uintptr(p)
}
