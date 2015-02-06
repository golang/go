// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the #cgo CFLAGS directive works,
// with and without platform filters.
// See http://golang.org/issue/5224 for details.
package cgotest

/*
#cgo CFLAGS: -DCOMMON_VALUE=123
#cgo windows CFLAGS: -DIS_WINDOWS=1
#cgo !windows CFLAGS: -DIS_WINDOWS=0
int common = COMMON_VALUE;
int is_windows = IS_WINDOWS;
*/
import "C"

import (
	"runtime"
	"testing"
)

func testCflags(t *testing.T) {
	is_windows := C.is_windows == 1
	if is_windows != (runtime.GOOS == "windows") {
		t.Errorf("is_windows: %v, runtime.GOOS: %s", is_windows, runtime.GOOS)
	}
	if C.common != 123 {
		t.Errorf("common: %v (expected 123)", C.common)
	}
}
