// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import (
	"testing"
	"unsafe"
)

// extern void f7665(void);
import "C"

//export f7665
func f7665() {}

var bad7665 unsafe.Pointer = C.f7665
var good7665 uintptr = uintptr(C.f7665)

func test7665(t *testing.T) {
	if bad7665 == nil || uintptr(bad7665) != good7665 {
		t.Errorf("ptrs = %p, %#x, want same non-nil pointer", bad7665, good7665)
	}
}
