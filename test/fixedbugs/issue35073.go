// run -gcflags=-d=checkptr

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that reflect.Value.UnsafeAddr/Pointer is handled
// correctly by -d=checkptr

package main

import (
	"reflect"
	"unsafe"
)

func main() {
	n := 10
	m := make(map[string]string)

	_ = unsafe.Pointer(reflect.ValueOf(&n).Elem().UnsafeAddr())
	_ = unsafe.Pointer(reflect.ValueOf(&m).Elem().Pointer())
}
