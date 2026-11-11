// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || wasm || arm64)

package simd_test

import (
	"reflect"
	"simd/archsimd"
	"testing"
	"unsafe"
)

// TestLayout checks that the type checker, which computes unsafe.Sizeof,
// Alignof and Offsetof, agrees with the compiler, which lays out the types and
// reports their layout to reflect.
func TestLayout(t *testing.T) {
	type S struct {
		a int32
		v archsimd.Float32x4
		b int8
		m archsimd.Mask8x16
	}
	var s S
	checkLayout(t, unsafe.Sizeof(s), unsafe.Alignof(s),
		[]uintptr{unsafe.Offsetof(s.a), unsafe.Offsetof(s.v), unsafe.Offsetof(s.b), unsafe.Offsetof(s.m)},
		reflect.TypeFor[S]())
}

// checkLayout checks that a struct type's size, alignment and field offsets
// agree with what reflect reports for it.
func checkLayout(t *testing.T, size, align uintptr, offsets []uintptr, rt reflect.Type) {
	t.Helper()
	if size != rt.Size() || align != uintptr(rt.Align()) {
		t.Errorf("%v: unsafe size %d align %d, reflect size %d align %d", rt, size, align, rt.Size(), rt.Align())
	}
	for i, off := range offsets {
		if f := rt.Field(i); off != f.Offset {
			t.Errorf("%v.%s: unsafe offset %d, reflect offset %d", rt, f.Name, off, f.Offset)
		}
	}
}
