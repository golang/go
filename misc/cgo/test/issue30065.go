// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Don't make a private copy of an array when taking the address of an
// element.

package cgotest

// #include <string.h>
import "C"

import (
	"testing"
	"unsafe"
)

func test30065(t *testing.T) {
	var a [256]byte
	b := []byte("a")
	C.memcpy(unsafe.Pointer(&a), unsafe.Pointer(&b[0]), 1)
	if a[0] != 'a' {
		t.Errorf("&a failed: got %c, want %c", a[0], 'a')
	}

	b = []byte("b")
	C.memcpy(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), 1)
	if a[0] != 'b' {
		t.Errorf("&a[0] failed: got %c, want %c", a[0], 'b')
	}

	d := make([]byte, 256)
	b = []byte("c")
	C.memcpy(unsafe.Pointer(&d[0]), unsafe.Pointer(&b[0]), 1)
	if d[0] != 'c' {
		t.Errorf("&d[0] failed: got %c, want %c", d[0], 'c')
	}
}
