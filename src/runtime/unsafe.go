// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/math"
	"unsafe"
)

func unsafestring(ptr unsafe.Pointer, len int) {
	if len < 0 {
		panicunsafestringlen()
	}

	if uintptr(len) > -uintptr(ptr) {
		if ptr == nil {
			panicunsafestringnilptr()
		}
		panicunsafestringlen()
	}
}

// Keep this code in sync with cmd/compile/internal/walk/builtin.go:walkUnsafeString
func unsafestring64(ptr unsafe.Pointer, len64 int64) {
	len := int(len64)
	if int64(len) != len64 {
		panicunsafestringlen()
	}
	unsafestring(ptr, len)
}

func unsafestringcheckptr(ptr unsafe.Pointer, len64 int64) {
	unsafestring64(ptr, len64)

	// Check that underlying array doesn't straddle multiple heap objects.
	// unsafestring64 has already checked for overflow.
	if checkptrStraddles(ptr, uintptr(len64)) {
		throw("checkptr: unsafe.String result straddles multiple allocations")
	}
}

func panicunsafestringlen() {
	panic(errorString("unsafe.String: len out of range"))
}

func panicunsafestringnilptr() {
	panic(errorString("unsafe.String: ptr is nil and len is not zero"))
}

// Keep this code in sync with cmd/compile/internal/walk/builtin.go:walkUnsafeSlice
func unsafeslice(et *_type, ptr unsafe.Pointer, len int) {
	if len < 0 {
		panicunsafeslicelen()
	}

	if et.size == 0 {
		if ptr == nil && len > 0 {
			panicunsafeslicenilptr()
		}
	}

	mem, overflow := math.MulUintptr(et.size, uintptr(len))
	if overflow || mem > -uintptr(ptr) {
		if ptr == nil {
			panicunsafeslicenilptr()
		}
		panicunsafeslicelen()
	}
}

// Keep this code in sync with cmd/compile/internal/walk/builtin.go:walkUnsafeSlice
func unsafeslice64(et *_type, ptr unsafe.Pointer, len64 int64) {
	len := int(len64)
	if int64(len) != len64 {
		panicunsafeslicelen()
	}
	unsafeslice(et, ptr, len)
}

func unsafeslicecheckptr(et *_type, ptr unsafe.Pointer, len64 int64) {
	unsafeslice64(et, ptr, len64)

	// Check that underlying array doesn't straddle multiple heap objects.
	// unsafeslice64 has already checked for overflow.
	if checkptrStraddles(ptr, uintptr(len64)*et.size) {
		throw("checkptr: unsafe.Slice result straddles multiple allocations")
	}
}

func panicunsafeslicelen() {
	panic(errorString("unsafe.Slice: len out of range"))
}

func panicunsafeslicenilptr() {
	panic(errorString("unsafe.Slice: ptr is nil and len is not zero"))
}
