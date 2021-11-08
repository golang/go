// run -gcflags=-d=checkptr

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"unsafe"
)

var s []int

func main() {
	s = []int{42}
	h := (*reflect.SliceHeader)(unsafe.Pointer(&s))
	x := *(*int)(unsafe.Pointer(h.Data))
	if x != 42 {
		panic(x)
	}
}
