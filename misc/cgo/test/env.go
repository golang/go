// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <stdlib.h>
*/
import "C"
import (
	"os"
	"runtime"
	"testing"
	"unsafe"
)

// This is really an os package test but here for convenience.
func testSetEnv(t *testing.T) {
	if runtime.GOOS == "windows" {
		// Go uses SetEnvironmentVariable on windows. Howerver,
		// C runtime takes a *copy* at process startup of thei
		// OS environment, and stores it in environ/envp.
		// It is this copy that	getenv/putenv manipulate.
		t.Logf("skipping test")
		return
	}
	const key = "CGO_OS_TEST_KEY"
	const val = "CGO_OS_TEST_VALUE"
	os.Setenv(key, val)
	keyc := C.CString(key)
	defer C.free(unsafe.Pointer(keyc))
	v := C.getenv(keyc)
	if v == (*C.char)(unsafe.Pointer(uintptr(0))) {
		t.Fatal("getenv returned NULL")
	}
	vs := C.GoString(v)
	if vs != val {
		t.Fatalf("getenv() = %q; want %q", vs, val)
	}
}
