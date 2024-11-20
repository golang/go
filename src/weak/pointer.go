// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package weak

import (
	"internal/abi"
	"runtime"
	"unsafe"
)

// Pointer is a weak pointer to a value of type T.
//
// Two Pointer values compare equal if the pointers
// that they were created from compare equal. This property is retained even
// after the object referenced by the pointer used to create a weak reference
// is reclaimed.
//
// If multiple weak pointers are made to different offsets within same object
// (for example, pointers to different fields of the same struct), those pointers
// will not compare equal.
// If a weak pointer is created from an object that becomes unreachable, but is
// then resurrected due to a finalizer, that weak pointer will not compare equal
// with weak pointers created after resurrection.
//
// Calling Make with a nil pointer returns a weak pointer whose Value method
// always returns nil. The zero value of a Pointer behaves as if it was created
// by passing nil to Make and compares equal with such pointers.
type Pointer[T any] struct {
	u unsafe.Pointer
}

// Make creates a weak pointer from a strong pointer to some value of type T.
func Make[T any](ptr *T) Pointer[T] {
	// Explicitly force ptr to escape to the heap.
	ptr = abi.Escape(ptr)

	var u unsafe.Pointer
	if ptr != nil {
		u = runtime_registerWeakPointer(unsafe.Pointer(ptr))
	}
	runtime.KeepAlive(ptr)
	return Pointer[T]{u}
}

// Value returns the original pointer used to create the weak pointer.
// It returns nil if the value pointed to by the original pointer was reclaimed by
// the garbage collector.
// If a weak pointer points to an object with a finalizer, then Value will
// return nil as soon as the object's finalizer is queued for execution.
func (p Pointer[T]) Value() *T {
	return (*T)(runtime_makeStrongFromWeak(p.u))
}

// Implemented in runtime.

//go:linkname runtime_registerWeakPointer
func runtime_registerWeakPointer(unsafe.Pointer) unsafe.Pointer

//go:linkname runtime_makeStrongFromWeak
func runtime_makeStrongFromWeak(unsafe.Pointer) unsafe.Pointer
