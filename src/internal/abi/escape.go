// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

import "unsafe"

// NoEscape hides the pointer p from escape analysis, preventing it
// from escaping to the heap. It compiles down to nothing.
//
// WARNING: This is very subtle to use correctly. The caller must
// ensure that it's truly safe for p to not escape to the heap by
// maintaining runtime pointer invariants (for example, that globals
// and the heap may not generally point into a stack).
//
//go:nosplit
//go:nocheckptr
func NoEscape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

var alwaysFalse bool
var escapeSink any

// Escape forces any pointers in x to escape to the heap.
func Escape[T any](x T) T {
	if alwaysFalse {
		escapeSink = x
	}
	return x
}

// EscapeNonString forces v to be on the heap, if v contains a
// non-string pointer.
//
// This is used in hash/maphash.Comparable. We cannot hash pointers
// to local variables on stack, as their addresses might change on
// stack growth. Strings are okay as the hash depends on only the
// content, not the pointer.
//
// This is essentially
//
//	if hasNonStringPointers(T) { Escape(v) }
//
// Implemented as a compiler intrinsic.
func EscapeNonString[T any](v T) { panic("intrinsic") }

// EscapeToResultNonString models a data flow edge from v to the result,
// if v contains a non-string pointer. If v contains only string pointers,
// it returns a copy of v, but is not modeled as a data flow edge
// from the escape analysis's perspective.
//
// This is used in unique.clone, to model the data flow edge on the
// value with strings excluded, because strings are cloned (by
// content).
//
// TODO: probably we should define this as a intrinsic and EscapeNonString
// could just be "heap = EscapeToResultNonString(v)". This way we can model
// an edge to the result but not necessarily heap.
func EscapeToResultNonString[T any](v T) T {
	EscapeNonString(v)
	return *(*T)(NoEscape(unsafe.Pointer(&v)))
}
