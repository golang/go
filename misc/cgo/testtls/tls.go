// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotlstest

// #include <pthread.h>
// extern void setTLS(int);
// extern int getTLS();
import "C"

import (
	"runtime"
	"testing"
)

func testTLS(t *testing.T) {
	var keyVal C.int = 1234

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	C.setTLS(C.int(keyVal))
	storedVal := C.getTLS()

	if storedVal != keyVal {
		t.Fatalf("stored %d want %d", storedVal, keyVal)
	}
}
