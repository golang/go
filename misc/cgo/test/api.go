// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// API Compatibility Checks for cgo

package cgotest

// #include <stdlib.h>
// const char *api_hello = "hello!";
import "C"
import "unsafe"

func testAPI() {
	var cs *C.char
	cs = C.CString("hello")
	defer C.free(unsafe.Pointer(cs))
	var s string
	s = C.GoString((*C.char)(C.api_hello))
	s = C.GoStringN((*C.char)(C.api_hello), C.int(6))
	var b []byte
	b = C.GoBytes(unsafe.Pointer(C.api_hello), C.int(6))
	_, _ = s, b
}
