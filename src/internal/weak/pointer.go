// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The weak package is a package for managing weak pointers.

Weak pointers are pointers that explicitly do not keep a value live and
must be queried for a regular Go pointer.
The result of such a query may be observed as nil at any point after a
weakly-pointed-to object becomes eligible for reclamation by the garbage
collector.
More specifically, weak pointers become nil as soon as the garbage collector
identifies that the object is unreachable, before it is made reachable
again by a finalizer.
In terms of the C# language, these semantics are roughly equivalent to the
the semantics of "short" weak references.
In terms of the Java language, these semantics are roughly equivalent to the
semantics of the WeakReference type.

Using go:linkname to access this package and the functions it references
is explicitly forbidden by the toolchain because the semantics of this
package have not gone through the proposal process. By exposing this
functionality, we risk locking in the existing semantics due to Hyrum's Law.

If you believe you have a good use-case for weak references not already
covered by the standard library, file a proposal issue at
https://github.com/golang/go/issues instead of relying on this package.
*/
package weak

import (
	"internal/abi"
	"runtime"
	"unsafe"
)

// Pointer is a weak pointer to a value of type T.
//
// This value is comparable is guaranteed to compare equal if the pointers
// that they were created from compare equal. This property is retained even
// after the object referenced by the pointer used to create a weak reference
// is reclaimed.
//
// If multiple weak pointers are made to different offsets within same object
// (for example, pointers to different fields of the same struct), those pointers
// will not compare equal.
// If a weak pointer is created from an object that becomes reachable again due
// to a finalizer, that weak pointer will not compare equal with weak pointers
// created before it became unreachable.
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

// Strong creates a strong pointer from the weak pointer.
// Returns nil if the original value for the weak pointer was reclaimed by
// the garbage collector.
// If a weak pointer points to an object with a finalizer, then Strong will
// return nil as soon as the object's finalizer is queued for execution.
func (p Pointer[T]) Strong() *T {
	return (*T)(runtime_makeStrongFromWeak(p.u))
}

// Implemented in runtime.

//go:linkname runtime_registerWeakPointer
func runtime_registerWeakPointer(unsafe.Pointer) unsafe.Pointer

//go:linkname runtime_makeStrongFromWeak
func runtime_makeStrongFromWeak(unsafe.Pointer) unsafe.Pointer
