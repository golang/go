// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The lru package implements a fixed-size in-memory LRU cache.
package lru

import (
	"container/heap"
	"fmt"
	"sync"
)

type any = interface{} // TODO: remove once gopls only builds at go1.18+

// A Cache is a fixed-size in-memory LRU cache.
type Cache struct {
	capacity int

	mu    sync.Mutex
	used  int            // used capacity, in user-specified units
	m     map[any]*entry // k/v lookup
	lru   queue          // min-atime priority queue of *entry
	clock int64          // clock time, incremented whenever the cache is updated
}

type entry struct {
	key   any
	value any
	size  int   // caller-specified size
	atime int64 // last access / set time
	index int   // index of entry in the heap slice
}

// New creates a new Cache with the given capacity, which must be positive.
//
// The cache capacity uses arbitrary units, which are specified during the Set
// operation.
func New(capacity int) *Cache {
	if capacity == 0 {
		panic("zero capacity")
	}

	return &Cache{
		capacity: capacity,
		m:        make(map[any]*entry),
	}
}

// Get retrieves the value for the specified key, or nil if the key is not
// found.
//
// If the key is found, its access time is updated.
func (c *Cache) Get(key any) any {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.clock++ // every access updates the clock

	if e, ok := c.m[key]; ok { // cache hit
		e.atime = c.clock
		heap.Fix(&c.lru, e.index)
		return e.value
	}

	return nil
}

// Set stores a value for the specified key, using its given size to update the
// current cache size, evicting old entries as necessary to fit in the cache
// capacity.
//
// Size must be a non-negative value. If size is larger than the cache
// capacity, the value is not stored and the cache is not modified.
func (c *Cache) Set(key, value any, size int) {
	if size < 0 {
		panic(fmt.Sprintf("size must be non-negative, got %d", size))
	}
	if size > c.capacity {
		return // uncacheable
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.clock++

	// Remove the existing cache entry for key, if it exists.
	e, ok := c.m[key]
	if ok {
		c.used -= e.size
		heap.Remove(&c.lru, e.index)
		delete(c.m, key)
	}

	// Evict entries until the new value will fit.
	newUsed := c.used + size
	if newUsed < 0 {
		return // integer overflow; return silently
	}
	c.used = newUsed
	for c.used > c.capacity {
		// evict oldest entry
		e = heap.Pop(&c.lru).(*entry)
		c.used -= e.size
		delete(c.m, e.key)
	}

	// Store the new value.
	// Opt: e is evicted, so it can be reused to reduce allocation.
	if e == nil {
		e = new(entry)
	}
	e.key = key
	e.value = value
	e.size = size
	e.atime = c.clock
	c.m[e.key] = e
	heap.Push(&c.lru, e)

	if len(c.m) != len(c.lru) {
		panic("map and LRU are inconsistent")
	}
}

// -- priority queue boilerplate --

// queue is a min-atime priority queue of cache entries.
type queue []*entry

func (q queue) Len() int { return len(q) }

func (q queue) Less(i, j int) bool { return q[i].atime < q[j].atime }

func (q queue) Swap(i, j int) {
	q[i], q[j] = q[j], q[i]
	q[i].index = i
	q[j].index = j
}

func (q *queue) Push(x any) {
	e := x.(*entry)
	e.index = len(*q)
	*q = append(*q, e)
}

func (q *queue) Pop() any {
	last := len(*q) - 1
	e := (*q)[last]
	(*q)[last] = nil // aid GC
	*q = (*q)[:last]
	return e
}
