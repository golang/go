// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
char* greeting = "hello, world";
#cgo darwin/amd64 LDFLAGS: -lregexp
#cgo linux CFLAGS: -m32
#cgo pkg-config: cairo moscow
*/
// #include "cgotest.h"
import "C"
import "unsafe"

var Greeting = C.GoString(C.greeting)

func DoAdd(x, y int) (sum int) {
	C.Add(C.int(x), C.int(y), (*C.int)(unsafe.Pointer(&sum)))
	return
}
