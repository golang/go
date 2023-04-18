// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

// NOTE: keep in sync with cmd/compile/internal/types.CalcSize
// to make the compiler recognize this as an intrinsic type.
type nih struct{}

// NotInHeap is a type must never be allocated from the GC'd heap or on the stack,
// and is called not-in-heap.
//
// Other types can embed NotInHeap to make it not-in-heap. Specifically, pointers
// to these types must always fail the `runtime.inheap` check. The type may be used
// for global variables, or for objects in unmanaged memory (e.g., allocated with
// `sysAlloc`, `persistentalloc`, r`fixalloc`, or from a manually-managed span).
//
// Specifically:
//
// 1. `new(T)`, `make([]T)`, `append([]T, ...)` and implicit heap
// allocation of T are disallowed. (Though implicit allocations are
// disallowed in the runtime anyway.)
//
// 2. A pointer to a regular type (other than `unsafe.Pointer`) cannot be
// converted to a pointer to a not-in-heap type, even if they have the
// same underlying type.
//
// 3. Any type that containing a not-in-heap type is itself considered as not-in-heap.
//
// - Structs and arrays are not-in-heap if their elements are not-in-heap.
// - Maps and channels contains no-in-heap types are disallowed.
//
// 4. Write barriers on pointers to not-in-heap types can be omitted.
//
// The last point is the real benefit of NotInHeap. The runtime uses
// it for low-level internal structures to avoid memory barriers in the
// scheduler and the memory allocator where they are illegal or simply
// inefficient. This mechanism is reasonably safe and does not compromise
// the readability of the runtime.
type NotInHeap struct{ _ nih }
