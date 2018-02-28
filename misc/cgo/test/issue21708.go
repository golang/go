// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// #include <stdint.h>
// #define CAST_TO_INT64 (int64_t)(-1)
import "C"
import "testing"

func test21708(t *testing.T) {
	if got, want := C.CAST_TO_INT64, -1; got != want {
		t.Errorf("C.CAST_TO_INT64 == %v, expected %v", got, want)
	}
}
