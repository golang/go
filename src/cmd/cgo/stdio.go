// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// #include <stdio.h>
// #include <stdlib.h>
import "C"

type File C.FILE;

func (f *File) Putc(c int) {
	C.putc(C.int(c), (*C.FILE)(f));
}

func (f *File) Puts(s string) {
	p := C.CString(s);
	C.fputs(p, (*C.FILE)(f));
	C.free(unsafe.Pointer(p));
}

var Stdout = (*File)(C.stdout);

func main() {
	Stdout.Puts("hello, world\n");
}
