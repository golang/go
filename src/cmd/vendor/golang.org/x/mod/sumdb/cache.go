// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parallel cache.
// This file is copied from cmd/go/internal/par.

package sumdb

import (
	"sync"
	"sync/atomic"
)

// parCache runs an action once per key and caches the result.
type parCache struct {
	m sync.Map
}

type cacheEntry struct {
	done   uint32
	mu     sync.Mutex
	result any
}

// Do calls the function f if and only if Do is being called for the first time with this key.
// No call to Do with a given key returns until the one call to f returns.
// Do returns the value returned by the one call to f.
func (c *parCache) Do(key any, f func() any) any {
	entryIface, ok := c.m.Load(key)
	if !ok {
		entryIface, _ = c.m.LoadOrStore(key, new(cacheEntry))
	}
	e := entryIface.(*cacheEntry)
	if atomic.LoadUint32(&e.done) == 0 {
		e.mu.Lock()
		if atomic.LoadUint32(&e.done) == 0 {
			e.result = f()
			atomic.StoreUint32(&e.done, 1)
		}
		e.mu.Unlock()
	}
	return e.result
}

// Get returns the cached result associated with key.
// It returns nil if there is no such result.
// If the result for key is being computed, Get does not wait for the computation to finish.
func (c *parCache) Get(key any) any {
	entryIface, ok := c.m.Load(key)
	if !ok {
		return nil
	}
	e := entryIface.(*cacheEntry)
	if atomic.LoadUint32(&e.done) == 0 {
		return nil
	}
	return e.result
}
