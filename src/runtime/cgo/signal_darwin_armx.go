// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin
// +build arm arm64

package cgo

import "unsafe"

//go:cgo_import_static x_cgo_panicmem
//go:linkname x_cgo_panicmem x_cgo_panicmem
var x_cgo_panicmem uintptr

// use a pointer to avoid relocation of external symbol in __TEXT
// make linker happy
var _cgo_panicmem = &x_cgo_panicmem

// TODO(crawshaw): move this into x_cgo_init, it will not run until
// runtime has finished loading, which may be after its use.
func init() {
	*_cgo_panicmem = funcPC(panicmem)
}

func funcPC(f interface{}) uintptr {
	var ptrSize = unsafe.Sizeof(uintptr(0))
	return **(**uintptr)(add(unsafe.Pointer(&f), ptrSize))
}

func add(p unsafe.Pointer, x uintptr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + x)
}

func panicmem()
