// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
A trivial example of wrapping a C library in Go.
For a more complex example and explanation,
see misc/cgo/gmp/gmp.go.
*/

package stdio

/*
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>

char* greeting = "hello, world";
*/
import "C"
import "unsafe"

type File C.FILE

// Test reference to library symbol.
// Stdout and stderr are too special to be a reliable test.
//var  = C.environ

func (f *File) WriteString(s string) {
	p := C.CString(s)
	C.fputs(p, (*C.FILE)(f))
	C.free(unsafe.Pointer(p))
	f.Flush()
}

func (f *File) Flush() {
	C.fflush((*C.FILE)(f))
}

var Greeting = C.GoString(C.greeting)
var Gbytes = C.GoBytes(unsafe.Pointer(C.greeting), C.int(len(Greeting)))
