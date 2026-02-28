// errorcheck -0 -d=checkptr -m

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we can inline the receiver arguments for
// reflect.Value.UnsafeAddr/Pointer, even in checkptr mode.

package main

import (
	"reflect"
	"unsafe"
)

func main() {
	n := 10                      // ERROR "moved to heap: n"
	m := make(map[string]string) // ERROR "moved to heap: m" "make\(map\[string\]string\) escapes to heap"

	_ = unsafe.Pointer(reflect.ValueOf(&n).Elem().UnsafeAddr()) // ERROR "inlining call"
	_ = unsafe.Pointer(reflect.ValueOf(&m).Elem().Pointer())    // ERROR "inlining call"
}
