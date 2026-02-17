// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package cgotest

/*
#cgo LDFLAGS: -Wl,-undefined,dynamic_lookup

extern void __gotest_cgo_null_api(void) __attribute__((weak_import));

int issue76023(void) {
    if (__gotest_cgo_null_api) return 1;
    return 0;
}
*/
import "C"
import "testing"

func issue76023(t *testing.T) {
	r := C.issue76023()
	if r != 0 {
		t.Error("found __gotest_cgo_null_api")
	}
}
