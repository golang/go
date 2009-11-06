// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
A trivial example of wrapping a C library in Go.
For a more complex example and explanation,
see ../gmp/gmp.go.
*/

package stdio

// TODO(rsc): Remove fflushstdout when C.fflush(C.stdout) works in cgo.

/*
#include <stdio.h>
#include <stdlib.h>

void fflushstdout(void) { fflush(stdout); }
*/
import "C"
import "unsafe"

/*
type File C.FILE

var Stdout = (*File)(C.stdout)
var Stderr = (*File)(C.stderr)

func (f *File) WriteString(s string) {
	p := C.CString(s);
	C.fputs(p, (*C.FILE)(f));
	C.free(p);
}
*/

func Puts(s string) {
	p := C.CString(s);
	C.puts(p);
	C.free(unsafe.Pointer(p));
	C.fflushstdout();
}
