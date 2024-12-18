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
// Just like regular pointers, Pointer may reference any part of an
// object, such as a field of a struct or an element of an array.
// Objects that are only pointed to by weak pointers are not considered
// reachable, and once the object becomes unreachable, [Pointer.Value]
// may return nil.
//
// The primary use-cases for weak pointers are for implementing caches,
// canonicalization maps (like the unique package), and for tying together
// the lifetimes of separate values (for example, through a map with weak
// keys).
//
// Two Pointer values always compare equal if the pointers from which they were
// created compare equal. This property is retained even after the
// object referenced by the pointer used to create a weak reference is
// reclaimed.
// If multiple weak pointers are made to different offsets within the same object
// (for example, pointers to different fields of the same struct), those pointers
// will not compare equal.
// If a weak pointer is created from an object that becomes unreachable, but is
// then resurrected due to a finalizer, that weak pointer will not compare equal
// with weak pointers created after the resurrection.
//
// Calling [Make] with a nil pointer returns a weak pointer whose [Pointer.Value]
// always returns nil. The zero value of a Pointer behaves as if it were created
// by passing nil to [Make] and compares equal with such pointers.
//
// [Pointer.Value] is not guaranteed to eventually return nil.
// [Pointer.Value] may return nil as soon as the object becomes
// unreachable.
// Values stored in global variables, or that can be found by tracing
// pointers from a global variable, are reachable. A function argument or
// receiver may become unreachable at the last point where the function
// mentions it. To ensure [Pointer.Value] does not return nil,
// pass a pointer to the object to the [runtime.KeepAlive] function after
// the last point where the object must remain reachable.
//
// Note that because [Pointer.Value] is not guaranteed to eventually return
// nil, even after an object is no longer referenced, the runtime is allowed to
// perform a space-saving optimization that batches objects together in a single
// allocation slot. The weak pointer for an unreferenced object in such an
// allocation may never be called if it always exists in the same batch as a
// referenced object. Typically, this batching only happens for tiny
// (on the order of 16 bytes or less) and pointer-free objects.
type Pointer[T any] struct {
	u unsafe.Pointer
}

// Make creates a weak pointer from a pointer to some value of type T.
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
