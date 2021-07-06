// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgosotest

// This test verifies that Go can access C variables
// in shared object file via cgo.

/*
// intentionally write the same LDFLAGS differently
// to test correct handling of LDFLAGS.
#cgo windows CFLAGS: -DIMPORT_DLL
#cgo linux LDFLAGS: -L. -lcgosotest
#cgo dragonfly LDFLAGS: -L. -l cgosotest
#cgo freebsd LDFLAGS: -L. -l cgosotest
#cgo openbsd LDFLAGS: -L. -l cgosotest
#cgo solaris LDFLAGS: -L. -lcgosotest
#cgo netbsd LDFLAGS: -L. libcgosotest.so
#cgo darwin LDFLAGS: -L. libcgosotest.dylib
#cgo windows LDFLAGS: -L. libcgosotest.a
#cgo aix LDFLAGS: -L. -l cgosotest

#include "cgoso_c.h"

const char* getVar() {
	    return exported_var;
}
*/
import "C"

import "fmt"

func Test() {
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
