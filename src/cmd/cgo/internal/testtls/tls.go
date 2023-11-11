// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotlstest

// extern const char *checkTLS();
// extern void setTLS(int);
// extern int getTLS();
import "C"

import (
	"runtime"
	"testing"
)

func testTLS(t *testing.T) {
	if skip := C.checkTLS(); skip != nil {
		t.Skipf("%s", C.GoString(skip))
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if val := C.getTLS(); val != 0 {
		t.Fatalf("at start, C.getTLS() = %#x, want 0", val)
	}

	const keyVal = 0x1234
	C.setTLS(keyVal)
	if val := C.getTLS(); val != keyVal {
		t.Fatalf("at end, C.getTLS() = %#x, want %#x", val, keyVal)
	}
}
