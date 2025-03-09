// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && amd64

package scan

import (
	"internal/runtime/gc"
	"simd"
	"unsafe"
)

// ExpandAVX512 expands each bit in packed into f consecutive bits in unpacked,
// where f is the word size of objects in sizeClass.
//
// This is a testing entrypoint to the expanders used by scanSpanPacked*.
func ExpandAVX512(sizeClass int, packed *gc.ObjMask, unpacked *gc.PtrMask) {
	v1, v2 := gcExpandersAVX512[sizeClass](unsafe.Pointer(packed))
	v1.Store((*[8]uint64)(unsafe.Pointer(unpacked)))
	v2.Store((*[8]uint64)(unsafe.Pointer(uintptr(unsafe.Pointer(unpacked)) + 64)))
	simd.ClearAVXUpperBits()
}
