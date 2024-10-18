// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements a cache of method sets.

package typeutil

import (
	"go/types"
	"sync"
)

// A MethodSetCache records the method set of each type T for which
// MethodSet(T) is called so that repeat queries are fast.
// The zero value is a ready-to-use cache instance.
type MethodSetCache struct {
	mu     sync.Mutex
	named  map[*types.Named]struct{ value, pointer *types.MethodSet } // method sets for named N and *N
	others map[types.Type]*types.MethodSet                            // all other types
}

// MethodSet returns the method set of type T.  It is thread-safe.
//
// If cache is nil, this function is equivalent to types.NewMethodSet(T).
// Utility functions can thus expose an optional *MethodSetCache
// parameter to clients that care about performance.
func (cache *MethodSetCache) MethodSet(T types.Type) *types.MethodSet {
	if cache == nil {
		return types.NewMethodSet(T)
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()

	switch T := types.Unalias(T).(type) {
	case *types.Named:
		return cache.lookupNamed(T).value

	case *types.Pointer:
		if N, ok := types.Unalias(T.Elem()).(*types.Named); ok {
			return cache.lookupNamed(N).pointer
		}
	}

	// all other types
	// (The map uses pointer equivalence, not type identity.)
	mset := cache.others[T]
	if mset == nil {
		mset = types.NewMethodSet(T)
		if cache.others == nil {
			cache.others = make(map[types.Type]*types.MethodSet)
		}
		cache.others[T] = mset
	}
	return mset
}

func (cache *MethodSetCache) lookupNamed(named *types.Named) struct{ value, pointer *types.MethodSet } {
	if cache.named == nil {
		cache.named = make(map[*types.Named]struct{ value, pointer *types.MethodSet })
	}
	// Avoid recomputing mset(*T) for each distinct Pointer
	// instance whose underlying type is a named type.
	msets, ok := cache.named[named]
	if !ok {
		msets.value = types.NewMethodSet(named)
		msets.pointer = types.NewMethodSet(types.NewPointer(named))
		cache.named[named] = msets
	}
	return msets
}
