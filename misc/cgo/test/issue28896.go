// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cgo was incorrectly adding padding after a packed struct.

package cgotest

/*
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	void *f1;
	uint32_t f2;
} __attribute__((__packed__)) innerPacked;

typedef struct {
	innerPacked g1;
	uint64_t g2;
} outerPacked;

typedef struct {
	void *f1;
	uint32_t f2;
} innerUnpacked;

typedef struct {
	innerUnpacked g1;
	uint64_t g2;
} outerUnpacked;

size_t offset(int x) {
	switch (x) {
	case 0:
		return offsetof(innerPacked, f2);
	case 1:
		return offsetof(outerPacked, g2);
	case 2:
		return offsetof(innerUnpacked, f2);
	case 3:
		return offsetof(outerUnpacked, g2);
	default:
		abort();
	}
}
*/
import "C"

import (
	"testing"
	"unsafe"
)

func offset(i int) uintptr {
	var pi C.innerPacked
	var po C.outerPacked
	var ui C.innerUnpacked
	var uo C.outerUnpacked
	switch i {
	case 0:
		return unsafe.Offsetof(pi.f2)
	case 1:
		return unsafe.Offsetof(po.g2)
	case 2:
		return unsafe.Offsetof(ui.f2)
	case 3:
		return unsafe.Offsetof(uo.g2)
	default:
		panic("can't happen")
	}
}

func test28896(t *testing.T) {
	for i := 0; i < 4; i++ {
		c := uintptr(C.offset(C.int(i)))
		g := offset(i)
		if c != g {
			t.Errorf("%d: C: %d != Go %d", i, c, g)
		}
	}
}
