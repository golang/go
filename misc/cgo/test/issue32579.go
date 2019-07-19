// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// #include <string.h>
// typedef struct S32579 { unsigned char data[1]; } S32579;
import "C"

import (
	"testing"
	"unsafe"
)

func test32579(t *testing.T) {
	var s [1]C.struct_S32579
	C.memset(unsafe.Pointer(&s[0].data[0]), 1, 1)
	if s[0].data[0] != 1 {
		t.Errorf("&s[0].data[0] failed: got %d, want %d", s[0].data[0], 1)
	}
}
