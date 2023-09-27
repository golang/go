// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package reflect_test

import (
	. "reflect"
	"runtime/cgo"
	"testing"
	"unsafe"
)

type nih struct {
	_ cgo.Incomplete
	x int
}

var global_nih = nih{x: 7}

func TestNotInHeapDeref(t *testing.T) {
	// See issue 48399.
	v := ValueOf((*nih)(nil))
	v.Elem()
	shouldPanic("reflect: call of reflect.Value.Field on zero Value", func() { v.Elem().Field(0) })

	v = ValueOf(&global_nih)
	if got := v.Elem().Field(1).Int(); got != 7 {
		t.Fatalf("got %d, want 7", got)
	}

	v = ValueOf((*nih)(unsafe.Pointer(new(int))))
	shouldPanic("reflect: reflect.Value.Elem on an invalid notinheap pointer", func() { v.Elem() })
	shouldPanic("reflect: reflect.Value.Pointer on an invalid notinheap pointer", func() { v.Pointer() })
	shouldPanic("reflect: reflect.Value.UnsafePointer on an invalid notinheap pointer", func() { v.UnsafePointer() })
}
