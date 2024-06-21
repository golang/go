// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique

import (
	"internal/abi"
	isync "internal/sync"
	"internal/weak"
	"runtime"
	"sync"
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
		// This is a good time to initialize cleanup, since we must go through
		// this path on the first use of Make, and it's not on the hot path.
		setupMake.Do(registerCleanup)
		ma = addUniqueMap[T](typ)
	}
	m := ma.(*uniqueMap[T])

	// Keep around any values we allocate for insertion. There
	// are a few different ways we can race with other threads
	// and create values that we might discard. By keeping
	// the first one we make around, we can avoid generating
	// more than one per racing thread.
	var (
		toInsert     *T // Keep this around to keep it alive.
		toInsertWeak weak.Pointer[T]
	)
	newValue := func() (T, weak.Pointer[T]) {
		if toInsert == nil {
			toInsert = new(T)
			*toInsert = clone(value, &m.cloneSeq)
			toInsertWeak = weak.Make(toInsert)
		}
		return *toInsert, toInsertWeak
	}
	var ptr *T
	for {
		// Check the map.
		wp, ok := m.Load(value)
		if !ok {
			// Try to insert a new value into the map.
			k, v := newValue()
			wp, _ = m.LoadOrStore(k, v)
		}
		// Now that we're sure there's a value in the map, let's
		// try to get the pointer we need out of it.
		ptr = wp.Strong()
		if ptr != nil {
			break
		}
		// The weak pointer is nil, so the old value is truly dead.
		// Try to remove it and start over.
		m.CompareAndDelete(value, wp)
	}
	runtime.KeepAlive(toInsert)
	return Handle[T]{ptr}
}

var (
	// uniqueMaps is an index of type-specific sync maps used for unique.Make.
	//
	// The two-level map might seem odd at first since the HashTrieMap could have "any"
	// as its key type, but the issue is escape analysis. We do not want to force lookups
	// to escape the argument, and using a type-specific map allows us to avoid that where
	// possible (for example, for strings and plain-ol'-data structs). We also get the
	// benefit of not cramming every different type into a single map, but that's certainly
	// not enough to outweigh the cost of two map lookups. What is worth it though, is saving
	// on those allocations.
	uniqueMaps = isync.NewHashTrieMap[*abi.Type, any]() // any is always a *uniqueMap[T].

	// cleanupFuncs are functions that clean up dead weak pointers in type-specific
	// maps in uniqueMaps. We express cleanup this way because there's no way to iterate
	// over the sync.Map and call functions on the type-specific data structures otherwise.
	// These cleanup funcs each close over one of these type-specific maps.
	//
	// cleanupMu protects cleanupNotify and is held across the entire cleanup. Used for testing.
	// cleanupNotify is a test-only mechanism that allow tests to wait for the cleanup to run.
	cleanupMu      sync.Mutex
	cleanupFuncsMu sync.Mutex
	cleanupFuncs   []func()
	cleanupNotify  []func() // One-time notifications when cleanups finish.
)

type uniqueMap[T comparable] struct {
	*isync.HashTrieMap[T, weak.Pointer[T]]
	cloneSeq
}

func addUniqueMap[T comparable](typ *abi.Type) *uniqueMap[T] {
	// Create a map for T and try to register it. We could
	// race with someone else, but that's fine; it's one
	// small, stray allocation. The number of allocations
	// this can create is bounded by a small constant.
	m := &uniqueMap[T]{
		HashTrieMap: isync.NewHashTrieMap[T, weak.Pointer[T]](),
		cloneSeq:    makeCloneSeq(typ),
	}
	a, loaded := uniqueMaps.LoadOrStore(typ, m)
	if !loaded {
		// Add a cleanup function for the new map.
		cleanupFuncsMu.Lock()
		cleanupFuncs = append(cleanupFuncs, func() {
			// Delete all the entries whose weak references are nil and clean up
			// deleted entries.
			m.All()(func(key T, wp weak.Pointer[T]) bool {
				if wp.Strong() == nil {
					m.CompareAndDelete(key, wp)
				}
				return true
			})
		})
		cleanupFuncsMu.Unlock()
	}
	return a.(*uniqueMap[T])
}

// setupMake is used to perform initial setup for unique.Make.
var setupMake sync.Once

// startBackgroundCleanup sets up a background goroutine to occasionally call cleanupFuncs.
func registerCleanup() {
	runtime_registerUniqueMapCleanup(func() {
		// Lock for cleanup.
		cleanupMu.Lock()

		// Grab funcs to run.
		cleanupFuncsMu.Lock()
		cf := cleanupFuncs
		cleanupFuncsMu.Unlock()

		// Run cleanup.
		for _, f := range cf {
			f()
		}

		// Run cleanup notifications.
		for _, f := range cleanupNotify {
			f()
		}
		cleanupNotify = nil

		// Finished.
		cleanupMu.Unlock()
	})
}

// Implemented in runtime.

//go:linkname runtime_registerUniqueMapCleanup
func runtime_registerUniqueMapCleanup(cleanup func())
