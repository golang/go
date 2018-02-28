// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 18126: cgo check of void function returning errno.

package cgotest

/*
#include <stdlib.h>

void Issue18126C(void **p) {
}
*/
import "C"

import (
	"testing"
)

func test18126(t *testing.T) {
	p := C.malloc(1)
	_, err := C.Issue18126C(&p)
	C.free(p)
	_ = err
}
