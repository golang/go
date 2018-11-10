// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 12030. sprintf is defined in both ntdll and msvcrt,
// Normally we want the one in the msvcrt.

package cgotest

/*
#include <stdio.h>
#include <stdlib.h>
void issue12030conv(char *buf, double x) {
	sprintf(buf, "d=%g", x);
}
*/
import "C"

import (
	"fmt"
	"testing"
	"unsafe"
)

func test12030(t *testing.T) {
	buf := (*C.char)(C.malloc(256))
	defer C.free(unsafe.Pointer(buf))
	for _, f := range []float64{1.0, 2.0, 3.14} {
		C.issue12030conv(buf, C.double(f))
		got := C.GoString(buf)
		if want := fmt.Sprintf("d=%g", f); got != want {
			t.Fatalf("C.sprintf failed for %g: %q != %q", f, got, want)
		}
	}
}
