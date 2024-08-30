// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.arenas

/*
The arena package provides the ability to allocate memory for a collection
of Go values and free that space manually all at once, safely. The purpose
of this functionality is to improve efficiency: manually freeing memory
before a garbage collection delays that cycle. Less frequent cycles means
the CPU cost of the garbage collector is incurred less frequently.

This functionality in this package is mostly captured in the Arena type.
Arenas allocate large chunks of memory for Go values, so they're likely to
be inefficient for allocating only small amounts of small Go values. They're
best used in bulk, on the order of MiB of memory allocated on each use.

Note that by allowing for this limited form of manual memory allocation
that use-after-free bugs are possible with regular Go values. This package
limits the impact of these use-after-free bugs by preventing reuse of freed
memory regions until the garbage collector is able to determine that it is
safe. Typically, a use-after-free bug will result in a fault and a helpful
error message, but this package reserves the right to not force a fault on
freed memory. That means a valid implementation of this package is to just
allocate all memory the way the runtime normally would, and in fact, it
reserves the right to occasionally do so for some Go values.
*/
package arena

import (
	"internal/reflectlite"
	"unsafe"
)

// Arena represents a collection of Go values allocated and freed together.
// Arenas are useful for improving efficiency as they may be freed back to
// the runtime manually, though any memory obtained from freed arenas must
// not be accessed once that happens. An Arena is automatically freed once
// it is no longer referenced, so it must be kept alive (see runtime.KeepAlive)
// until any memory allocated from it is no longer needed.
//
// An Arena must never be used concurrently by multiple goroutines.
type Arena struct {
	a unsafe.Pointer
}

// NewArena allocates a new arena.
func NewArena() *Arena {
	return &Arena{a: runtime_arena_newArena()}
}

// Free frees the arena (and all objects allocated from the arena) so that
// memory backing the arena can be reused fairly quickly without garbage
// collection overhead. Applications must not call any method on this
// arena after it has been freed.
func (a *Arena) Free() {
	runtime_arena_arena_Free(a.a)
	a.a = nil
}

// New creates a new *T in the provided arena. The *T must not be used after
// the arena is freed. Accessing the value after free may result in a fault,
// but this fault is also not guaranteed.
func New[T any](a *Arena) *T {
	return runtime_arena_arena_New(a.a, reflectlite.TypeOf((*T)(nil))).(*T)
}

// MakeSlice creates a new []T with the provided capacity and length. The []T must
// not be used after the arena is freed. Accessing the underlying storage of the
// slice after free may result in a fault, but this fault is also not guaranteed.
func MakeSlice[T any](a *Arena, len, cap int) []T {
	var sl []T
	runtime_arena_arena_Slice(a.a, &sl, cap)
	return sl[:len]
}

// Clone makes a shallow copy of the input value that is no longer bound to any
// arena it may have been allocated from, returning the copy. If it was not
// allocated from an arena, it is returned untouched. This function is useful
// to more easily let an arena-allocated value out-live its arena.
// T must be a pointer, a slice, or a string, otherwise this function will panic.
func Clone[T any](s T) T {
	return runtime_arena_heapify(s).(T)
}

//go:linkname reflect_arena_New reflect.arena_New
func reflect_arena_New(a *Arena, typ any) any {
	return runtime_arena_arena_New(a.a, typ)
}

//go:linkname runtime_arena_newArena
func runtime_arena_newArena() unsafe.Pointer

//go:linkname runtime_arena_arena_New
func runtime_arena_arena_New(arena unsafe.Pointer, typ any) any

// Mark as noescape to avoid escaping the slice header.
//
//go:noescape
//go:linkname runtime_arena_arena_Slice
func runtime_arena_arena_Slice(arena unsafe.Pointer, slice any, cap int)

//go:linkname runtime_arena_arena_Free
func runtime_arena_arena_Free(arena unsafe.Pointer)

//go:linkname runtime_arena_heapify
func runtime_arena_heapify(any) any
