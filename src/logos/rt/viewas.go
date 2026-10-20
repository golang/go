// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"strconv"
	"unsafe"
)

// ViewAs reinterprets the leading bytes of b as a *T (SPEC T-205,
// M7-S5). T must be a POD struct: the compiler rejects padding and
// reference-carrying fields with [comptime::viewas_not_pod], so only
// the length check runs here. A short input panics with
// `logos: ViewAs: need <N> bytes, have <M>`.
//
// The returned pointer addresses b's backing array, so a live view
// keeps b alive through the collector's interior-pointer resolution;
// on architectures without unaligned loads the view addresses an
// aligned copy instead (viewas_copy.go).
func ViewAs[T any](b []byte) *T {
	var zero T
	size := int(unsafe.Sizeof(zero))
	if len(b) < size {
		panic("logos: ViewAs: need " + strconv.Itoa(size) + " bytes, have " + strconv.Itoa(len(b)))
	}
	return viewAsAddr[T](b)
}

// viewAsCopy reinterprets the leading bytes of b as a *T by copying
// them into a freshly allocated, naturally aligned T. It backs the
// fallback architectures (viewas_copy.go) and is exercised directly by
// unit tests on every platform. The caller checks the length: b must
// hold at least sizeof(T) bytes.
func viewAsCopy[T any](b []byte) *T {
	var zero T
	size := int(unsafe.Sizeof(zero))
	p := new(T)
	if size > 0 {
		copy(unsafe.Slice((*byte)(unsafe.Pointer(p)), size), b)
	}
	return p
}
