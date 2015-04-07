// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgosotest

/*
#cgo windows CFLAGS: -DIMPORT_DLL
// intentionally write the same LDFLAGS differently
// to test correct handling of LDFLAGS.
#cgo linux LDFLAGS: -L. -lcgosotest
#cgo dragonfly LDFLAGS: -L. -l cgosotest
#cgo freebsd LDFLAGS: -L. -l cgosotest
#cgo openbsd LDFLAGS: -L. -l cgosotest
#cgo netbsd LDFLAGS: -L. libcgosotest.so
#cgo darwin LDFLAGS: -L. libcgosotest.dylib
#cgo windows LDFLAGS: -L. libcgosotest.dll

#include "cgoso_c.h"

void init(void);
void sofunc(void);
const char* getVar(void);
*/
import "C"

import "fmt"

func Test() {
	C.init()
	C.sofunc()
	testExportedVar()
}

func testExportedVar() {
	const want = "Hello world"
	got := C.GoString(C.getVar())
	if got != want {
		panic(fmt.Sprintf("testExportedVar: got %q, but want %q", got, want))
	}
	got = C.GoString(C.exported_var)
	if got != want {
		panic(fmt.Sprintf("testExportedVar: got %q, but want %q", got, want))
	}
}

//export goCallback
func goCallback() {
}
