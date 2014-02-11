// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a cache of method sets.

package types

import "sync"

// A MethodSetCache records the method set of each type T for which
// MethodSet(T) is called so that repeat queries are fast.
// The zero value is a ready-to-use cache instance.
type MethodSetCache struct {
	mu     sync.Mutex
	named  map[*Named]struct{ value, pointer *MethodSet } // method sets for named N and *N
	others map[Type]*MethodSet                            // all other types
}

// MethodSet returns the method set of type T.  It is thread-safe.
//
// If cache is nil, this function is equivalent to NewMethodSet(T).
// Utility functions can thus expose an optional *MethodSetCache
// parameter to clients that care about performance.
//
func (cache *MethodSetCache) MethodSet(T Type) *MethodSet {
	if cache == nil {
		return NewMethodSet(T)
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()

	switch T := T.(type) {
	case *Named:
		return cache.lookupNamed(T).value

	case *Pointer:
		if N, ok := T.Elem().(*Named); ok {
			return cache.lookupNamed(N).pointer
		}
	}

	// all other types
	// (The map uses pointer equivalence, not type identity.)
	mset := cache.others[T]
	if mset == nil {
		mset = NewMethodSet(T)
		if cache.others == nil {
			cache.others = make(map[Type]*MethodSet)
		}
		cache.others[T] = mset
	}
	return mset
}

func (cache *MethodSetCache) lookupNamed(named *Named) struct{ value, pointer *MethodSet } {
	if cache.named == nil {
		cache.named = make(map[*Named]struct{ value, pointer *MethodSet })
	}
	// Avoid recomputing mset(*T) for each distinct Pointer
	// instance whose underlying type is a named type.
	msets, ok := cache.named[named]
	if !ok {
		msets.value = NewMethodSet(named)
		msets.pointer = NewMethodSet(NewPointer(named))
		cache.named[named] = msets
	}
	return msets
}
