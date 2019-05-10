// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package memoize supports functions with idempotent results that are expensive
// to compute having their return value memorized and returned again the next
// time they are invoked.
// The return values are only remembered for as long as there is still a user
// to prevent excessive memory use.
// To use this package, build a store and use it to aquire handles with the
// Bind method.
package memoize

import (
	"context"
	"runtime"
	"sync"
	"unsafe"
)

// Store binds keys to functions, returning handles that can be used to access
// the functions results.
type Store struct {
	mu sync.Mutex
	// entries is the set of values stored.
	entries map[interface{}]*entry
}

// Function is the type for functions that can be memoized.
// The result must be a pointer.
type Function func(ctx context.Context) interface{}

// Handle is returned from a store when a key is bound to a function.
// It is then used to access the results of that function.
type Handle struct {
	mu       sync.Mutex
	function Function
	entry    *entry
	value    interface{}
}

// entry holds the machinery to manage a function and its result such that
// there is only one instance of the result live at any given time.
type entry struct {
	noCopy
	// mu contols access to the typ and ptr fields
	mu sync.Mutex
	// the calculated value, as stored in an interface{}
	typ, ptr uintptr
	ready    bool
	// wait is used to block until the value is ready
	// will only be non nil if the generator is already running
	wait chan struct{}
}

// Has returns true if they key is currently valid for this store.
func (s *Store) Has(key interface{}) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, found := s.entries[key]
	return found
}

// Delete removes a key from the store, if present.
func (s *Store) Delete(key interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.entries, key)
}

// Bind returns a handle for the given key and function.
// Each call to bind will generate a new handle, but all the handles for a
// single key will refer to the same value, and only the first handle to try to
// get the value will cause the function to be invoked.
// The results of the function are held for as long as there are handles through
// which the result has been accessed.
// Bind does not cause the value to be generated.
func (s *Store) Bind(key interface{}, function Function) *Handle {
	// panic early if the function is nil
	// it would panic later anyway, but in a way that was much harder to debug
	if function == nil {
		panic("Function passed to bind must not be nil")
	}
	// check if we already have the key
	s.mu.Lock()
	defer s.mu.Unlock()
	e, found := s.entries[key]
	if !found {
		// we have not seen this key before, add a new entry
		if s.entries == nil {
			s.entries = make(map[interface{}]*entry)
		}
		e = &entry{}
		s.entries[key] = e
	}
	return &Handle{
		entry:    e,
		function: function,
	}
}

// Cached returns the value associated with a key.
// It cannot cause the value to be generated, but will return the cached
// value if present.
func (s *Store) Cached(key interface{}) interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, found := s.entries[key]
	if !found {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return unref(e)
}

// Cached returns the value associated with a handle.
// It will never cause the value to be generated, it will return the cached
// value if present.
func (h *Handle) Cached() interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.value == nil {
		h.entry.mu.Lock()
		defer h.entry.mu.Unlock()
		h.value = unref(h.entry)
	}
	return h.value
}

// Get returns the value associated with a handle.
// If the value is not yet ready, the underlying function will be invoked.
// This makes this handle active, it will remember the value for as long as
// it exists, and cause any other handles for the same key to also return the
// same value.
func (h *Handle) Get(ctx context.Context) interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.function != nil {
		if v, ok := h.entry.get(ctx, h.function); ok {
			h.value = v
			h.function = nil
			h.entry = nil
		}
	}
	return h.value
}

// get is the implementation of Get.
func (e *entry) get(ctx context.Context, f Function) (interface{}, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	// fast path if we already have a value
	if e.ready {
		return unref(e), true
	}
	// value is not ready, and we hold the lock
	// see if the value is already being calculated
	var value interface{}
	if e.wait == nil {
		e.wait = make(chan struct{})
		go func() {
			defer func() {
				close(e.wait)
				e.wait = nil
			}()
			// e is not locked here
			ctx := context.Background()
			value = f(ctx)
			// function is done, return to locked state so we can update the entry
			e.mu.Lock()
			defer e.mu.Unlock()
			setref(e, value)
		}()
	}
	// get a local copy of wait while we still hold the lock
	wait := e.wait
	e.mu.Unlock()
	// release the lock while we wait
	select {
	case <-wait:
		// we should now have a value
		e.mu.Lock()
		result := unref(e)
		// the keep alive makes sure value is not garbage collected before unref
		runtime.KeepAlive(value)
		return result, true
	case <-ctx.Done():
		// our context was cancelled
		e.mu.Lock()
		return nil, false
	}
}

// setref is called to store a value into an entry
// it must only be called when the lock is held
func setref(e *entry, value interface{}) interface{} {
	// this is only called when the entry lock is already held
	data := (*[2]uintptr)(unsafe.Pointer(&value))
	// store the value back to the entry as a weak reference
	e.typ, e.ptr = data[0], data[1]
	e.ready = true
	if e.ptr != 0 {
		// and arrange to clear the weak reference if the object is collected
		runtime.SetFinalizer(value, func(_ interface{}) {
			// clear the now invalid non pointer
			e.mu.Lock()
			defer e.mu.Unlock()
			e.typ, e.ptr = 0, 0
			e.ready = false
		})
	}
	return value
}

func unref(e *entry) interface{} {
	// this is only called when the entry lock is already held
	var v interface{}
	data := (*[2]uintptr)(unsafe.Pointer(&v))
	data[0], data[1] = e.typ, e.ptr
	return v
}
