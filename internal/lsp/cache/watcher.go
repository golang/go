// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"sync"
)

type watcher struct {
	id       uint64
	callback func()
}

type WatchMap struct {
	mu       sync.Mutex
	nextID   uint64
	watchers map[interface{}][]watcher
}

func NewWatchMap() *WatchMap {
	return &WatchMap{watchers: make(map[interface{}][]watcher)}
}
func (w *WatchMap) Watch(key interface{}, callback func()) func() {
	w.mu.Lock()
	defer w.mu.Unlock()
	id := w.nextID
	w.nextID++
	w.watchers[key] = append(w.watchers[key], watcher{
		id:       id,
		callback: callback,
	})
	return func() {
		// unwatch if invoked
		w.mu.Lock()
		defer w.mu.Unlock()
		// find and delete the watcher entry
		entries := w.watchers[key]
		for i, entry := range entries {
			if entry.id == id {
				// found it
				entries[i] = entries[len(entries)-1]
				entries = entries[:len(entries)-1]
			}
		}
	}
}

func (w *WatchMap) Notify(key interface{}) {
	// Make a copy of the watcher callbacks so we don't need to hold
	// the mutex during the callbacks (to avoid deadlocks).
	w.mu.Lock()
	entries := w.watchers[key]
	entriesCopy := make([]watcher, len(entries))
	copy(entriesCopy, entries)
	w.mu.Unlock()

	for _, entry := range entriesCopy {
		entry.callback()
	}
}
