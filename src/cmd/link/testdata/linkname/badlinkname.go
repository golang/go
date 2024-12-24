// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Existing pull linknames in the wild are allowed _for now_,
// for legacy reason. Test a function and a method.
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

func main() {
	println(rtype_String(noescape(nil)))
}
