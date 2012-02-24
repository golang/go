// compile

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test unsafe.Sizeof, unsafe.Alignof, and unsafe.Offsetof all return uintptr.

package main

import "unsafe"

type T struct {
	X int
}

var t T

func isUintptr(uintptr) {}

func main() {
	isUintptr(unsafe.Sizeof(t))
	isUintptr(unsafe.Alignof(t))
	isUintptr(unsafe.Offsetof(t.X))
}
