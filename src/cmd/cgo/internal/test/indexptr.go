// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <stdlib.h>

void setCharPtr(char **p, char *v) { *p = v; }
char *getCharPtr(char **p) { return *p; }
*/
import "C"
import (
	"testing"
	"unsafe"
)

type indexPtrHolder struct {
	arr [4]*C.char
	f   *C.char
}

func testIndexPtrDeref(t *testing.T) {
	var live []*C.char
	mk := func() *C.char {
		s := C.CString("cell")
		live = append(live, s)
		return s
	}
	defer func() {
		for _, s := range live {
			C.free(unsafe.Pointer(s))
		}
	}()
	nv := mk()

	pa := &[4]*C.char{mk(), mk(), mk(), mk()}
	C.setCharPtr(&(*pa)[0], nv)
	if got := C.getCharPtr(&(*pa)[0]); got != nv {
		t.Errorf("&(*pa)[0] store lost: got %q, want new", C.GoString(got))
	}

	pa2 := &[2][2]*C.char{{mk(), mk()}, {mk(), mk()}}
	C.setCharPtr(&(*pa2)[0][1], nv)
	if (*pa2)[0][1] != nv {
		t.Errorf("&(*pa2)[0][1] store lost")
	}

	var h indexPtrHolder
	h.arr = [4]*C.char{mk(), mk(), mk(), mk()}
	h.f = mk()
	C.setCharPtr(&h.arr[2], nv)
	if h.arr[2] != nv {
		t.Errorf("&h.arr[2] store lost")
	}
	C.setCharPtr(&(*(&h)).f, nv)
	if h.f != nv {
		t.Errorf("&(*h).f store lost")
	}

	sl := []*C.char{mk(), mk(), mk()}
	C.setCharPtr(&sl[1], nv)
	if sl[1] != nv {
		t.Errorf("&sl[1] store lost")
	}
}
