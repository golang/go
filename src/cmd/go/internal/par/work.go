// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package par implements parallel execution helpers.
package par

import (
	"errors"
	"math/rand"
	"sync"
	"sync/atomic"
)

// Work manages a set of work items to be executed in parallel, at most once each.
// The items in the set must all be valid map keys.
type Work[T comparable] struct {
	f       func(T) // function to run for each item
	running int     // total number of runners

	mu      sync.Mutex
	added   map[T]bool // items added to set
	todo    []T        // items yet to be run
	wait    sync.Cond  // wait when todo is empty
	waiting int        // number of runners waiting for todo
}

func (w *Work[T]) init() {
	if w.added == nil {
		w.added = make(map[T]bool)
	}
}

// Add adds item to the work set, if it hasn't already been added.
func (w *Work[T]) Add(item T) {
	w.mu.Lock()
	w.init()
	if !w.added[item] {
		w.added[item] = true
		w.todo = append(w.todo, item)
		if w.waiting > 0 {
			w.wait.Signal()
		}
	}
	w.mu.Unlock()
}

// Do runs f in parallel on items from the work set,
// with at most n invocations of f running at a time.
// It returns when everything added to the work set has been processed.
// At least one item should have been added to the work set
// before calling Do (or else Do returns immediately),
// but it is allowed for f(item) to add new items to the set.
// Do should only be used once on a given Work.
func (w *Work[T]) Do(n int, f func(item T)) {
	if n < 1 {
		panic("par.Work.Do: n < 1")
	}
	if w.running >= 1 {
		panic("par.Work.Do: already called Do")
	}

	w.running = n
	w.f = f
	w.wait.L = &w.mu

	for i := 0; i < n-1; i++ {
		go w.runner()
	}
	w.runner()
}

// runner executes work in w until both nothing is left to do
// and all the runners are waiting for work.
// (Then all the runners return.)
func (w *Work[T]) runner() {
	for {
		// Wait for something to do.
		w.mu.Lock()
		for len(w.todo) == 0 {
			w.waiting++
			if w.waiting == w.running {
				// All done.
				w.wait.Broadcast()
				w.mu.Unlock()
				return
			}
			w.wait.Wait()
			w.waiting--
		}

		// Pick something to do at random,
		// to eliminate pathological contention
		// in case items added at about the same time
		// are most likely to contend.
		i := rand.Intn(len(w.todo))
		item := w.todo[i]
		w.todo[i] = w.todo[len(w.todo)-1]
		w.todo = w.todo[:len(w.todo)-1]
		w.mu.Unlock()

		w.f(item)
	}
}

// ErrCache is like Cache except that it also stores
// an error value alongside the cached value V.
type ErrCache[K comparable, V any] struct {
	Cache[K, errValue[V]]
}

type errValue[V any] struct {
	v   V
	err error
}

func (c *ErrCache[K, V]) Do(key K, f func() (V, error)) (V, error) {
	v := c.Cache.Do(key, func() errValue[V] {
		v, err := f()
		return errValue[V]{v, err}
	})
	return v.v, v.err
}

var ErrCacheEntryNotFound = errors.New("cache entry not found")

// Get returns the cached result associated with key.
// It returns ErrCacheEntryNotFound if there is no such result.
func (c *ErrCache[K, V]) Get(key K) (V, error) {
	v, ok := c.Cache.Get(key)
	if !ok {
		v.err = ErrCacheEntryNotFound
	}
	return v.v, v.err
}

// Cache runs an action once per key and caches the result.
type Cache[K comparable, V any] struct {
	m sync.Map
}

type cacheEntry[V any] struct {
	done   atomic.Bool
	mu     sync.Mutex
	result V
}

// Do calls the function f if and only if Do is being called for the first time with this key.
// No call to Do with a given key returns until the one call to f returns.
// Do returns the value returned by the one call to f.
func (c *Cache[K, V]) Do(key K, f func() V) V {
	entryIface, ok := c.m.Load(key)
	if !ok {
		entryIface, _ = c.m.LoadOrStore(key, new(cacheEntry[V]))
	}
	e := entryIface.(*cacheEntry[V])
	if !e.done.Load() {
		e.mu.Lock()
		if !e.done.Load() {
			e.result = f()
			e.done.Store(true)
		}
		e.mu.Unlock()
	}
	return e.result
}

// Get returns the cached result associated with key
// and reports whether there is such a result.
//
// If the result for key is being computed, Get does not wait for the computation to finish.
func (c *Cache[K, V]) Get(key K) (V, bool) {
	entryIface, ok := c.m.Load(key)
	if !ok {
		return *new(V), false
	}
	e := entryIface.(*cacheEntry[V])
	if !e.done.Load() {
		return *new(V), false
	}
	return e.result, true
}

// Clear removes all entries in the cache.
//
// Concurrent calls to Get may return old values. Concurrent calls to Do
// may return old values or store results in entries that have been deleted.
//
// TODO(jayconrod): Delete this after the package cache clearing functions
// in internal/load have been removed.
func (c *Cache[K, V]) Clear() {
	c.m.Range(func(key, value any) bool {
		c.m.Delete(key)
		return true
	})
}

// Delete removes an entry from the map. It is safe to call Delete for an
// entry that does not exist. Delete will return quickly, even if the result
// for a key is still being computed; the computation will finish, but the
// result won't be accessible through the cache.
//
// TODO(jayconrod): Delete this after the package cache clearing functions
// in internal/load have been removed.
func (c *Cache[K, V]) Delete(key K) {
	c.m.Delete(key)
}

// DeleteIf calls pred for each key in the map. If pred returns true for a key,
// DeleteIf removes the corresponding entry. If the result for a key is
// still being computed, DeleteIf will remove the entry without waiting for
// the computation to finish. The result won't be accessible through the cache.
//
// TODO(jayconrod): Delete this after the package cache clearing functions
// in internal/load have been removed.
func (c *Cache[K, V]) DeleteIf(pred func(key K) bool) {
	c.m.Range(func(key, _ any) bool {
		if key := key.(K); pred(key) {
			c.Delete(key)
		}
		return true
	})
}
