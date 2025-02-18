// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique

import (
	"internal/abi"
	isync "internal/sync"
	"unsafe"
)

var zero uintptr

// Handle is a globally unique identity for some value of type T.
//
// Two handles compare equal exactly if the two values used to create the handles
// would have also compared equal. The comparison of two handles is trivial and
// typically much more efficient than comparing the values used to create them.
type Handle[T comparable] struct {
	value *T
}

// Value returns a shallow copy of the T value that produced the Handle.
// Value is safe for concurrent use by multiple goroutines.
func (h Handle[T]) Value() T {
	return *h.value
}

// Make returns a globally unique handle for a value of type T. Handles
// are equal if and only if the values used to produce them are equal.
// Make is safe for concurrent use by multiple goroutines.
func Make[T comparable](value T) Handle[T] {
	// Find the map for type T.
	typ := abi.TypeFor[T]()
	if typ.Size() == 0 {
		return Handle[T]{(*T)(unsafe.Pointer(&zero))}
	}
	ma, ok := uniqueMaps.Load(typ)
	if !ok {
		m := &uniqueMap[T]{canonMap: newCanonMap[T](), cloneSeq: makeCloneSeq(typ)}
		ma, _ = uniqueMaps.LoadOrStore(typ, m)
	}
	m := ma.(*uniqueMap[T])

	// Find the value in the map.
	ptr := m.Load(value)
	if ptr == nil {
		// Insert a new value into the map.
		ptr = m.LoadOrStore(clone(value, &m.cloneSeq))
	}
	return Handle[T]{ptr}
}

// uniqueMaps is an index of type-specific concurrent maps used for unique.Make.
//
// The two-level map might seem odd at first since the HashTrieMap could have "any"
// as its key type, but the issue is escape analysis. We do not want to force lookups
// to escape the argument, and using a type-specific map allows us to avoid that where
// possible (for example, for strings and plain-ol'-data structs). We also get the
// benefit of not cramming every different type into a single map, but that's certainly
// not enough to outweigh the cost of two map lookups. What is worth it though, is saving
// on those allocations.
var uniqueMaps isync.HashTrieMap[*abi.Type, any] // any is always a *uniqueMap[T].

type uniqueMap[T comparable] struct {
	*canonMap[T]
	cloneSeq
}

// Implemented in runtime.
//
// Used only by tests.
//
//go:linkname runtime_blockUntilEmptyFinalizerQueue
func runtime_blockUntilEmptyFinalizerQueue(timeout int64) bool
