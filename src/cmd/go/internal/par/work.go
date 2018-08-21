// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package par implements parallel execution helpers.
package par

import (
	"math/rand"
	"sync"
	"sync/atomic"
)

// Work manages a set of work items to be executed in parallel, at most once each.
// The items in the set must all be valid map keys.
type Work struct {
	f       func(interface{}) // function to run for each item
	running int               // total number of runners

	mu      sync.Mutex
	added   map[interface{}]bool // items added to set
	todo    []interface{}        // items yet to be run
	wait    sync.Cond            // wait when todo is empty
	waiting int                  // number of runners waiting for todo
}

func (w *Work) init() {
	if w.added == nil {
		w.added = make(map[interface{}]bool)
	}
}

// Add adds item to the work set, if it hasn't already been added.
func (w *Work) Add(item interface{}) {
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
func (w *Work) Do(n int, f func(item interface{})) {
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
func (w *Work) runner() {
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

// Cache runs an action once per key and caches the result.
type Cache struct {
	m sync.Map
}

type cacheEntry struct {
	done   uint32
	mu     sync.Mutex
	result interface{}
}

// Do calls the function f if and only if Do is being called for the first time with this key.
// No call to Do with a given key returns until the one call to f returns.
// Do returns the value returned by the one call to f.
func (c *Cache) Do(key interface{}, f func() interface{}) interface{} {
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
func (c *Cache) Get(key interface{}) interface{} {
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
