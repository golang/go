// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Existing pull linknames in the wild are allowed _for now_,
// for legacy reason. Test a function, a method, and an
// assembly symbol.
// NOTE: this may not be allowed in the future. Don't do this!

package main

import (
	_ "reflect"
	"unsafe"
)

//go:linkname noescape runtime.noescape
func noescape(unsafe.Pointer) unsafe.Pointer

//go:linkname rtype_String reflect.(*rtype).String
func rtype_String(unsafe.Pointer) string

//go:linkname memmove runtime.memmove
func memmove(to, from unsafe.Pointer, n uintptr)

var n uintptr // use a global to prevent compiler optimize out memmove call

func main() {
	println(rtype_String(noescape(nil)))
	memmove(nil, nil, n)
}
