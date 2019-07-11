// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package memoize supports memoizing the return values of functions with
// idempotent results that are expensive to compute.
//
// The memoizied result is returned again the next time the function is invoked.
// To prevent excessive memory use, the return values are only remembered
// for as long as they still have a user.
//
// To use this package, build a store and use it to aquire handles with the
// Bind method.
//
package memoize

import (
	"context"
	"runtime"
	"sync"
	"unsafe"

	"golang.org/x/tools/internal/xcontext"
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
	key interface{}
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
//
// Each call to bind will generate a new handle.
// All of of the handles for a single key will refer to the same value.
// Only the first handle to get the value will cause the function to be invoked.
// The value will be held for as long as there are handles through which it has been accessed.
// Bind does not cause the value to be generated.
func (s *Store) Bind(key interface{}, function Function) *Handle {
	// panic early if the function is nil
	// it would panic later anyway, but in a way that was much harder to debug
	if function == nil {
		panic("the function passed to bind must not be nil")
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
		e = &entry{key: key}
		s.entries[key] = e
	}
	return &Handle{
		entry:    e,
		function: function,
	}
}

// Cached returns the value associated with a key.
//
// It cannot cause the value to be generated.
// It will return the cached value, if present.
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
//
// It will never cause the value to be generated.
// It will return the cached value, if present.
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
//
// If the value is not yet ready, the underlying function will be invoked.
// This activates the handle, and it will remember the value for as long as it exists.
// This will cause any other handles for the same key to also return the same value.
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
	// Note: This defer is not paired with the above lock.
	defer e.mu.Unlock()

	// Fast path: If the entry is ready, it already has a value.
	if e.ready {
		return unref(e), true
	}
	// Only begin evaluating the function if no other goroutine is doing so.
	var value interface{}
	if e.wait == nil {
		e.wait = make(chan struct{})
		go func() {
			// Note: We do not hold the lock on the entry in this goroutine.
			//
			// We immediately defer signaling that the entry is ready,
			// since we cannot guarantee that the function, f, will not panic.
			defer func() {
				// Note: We have to hold the entry's lock before returning.
				close(e.wait)
				e.wait = nil
			}()

			// Use the background context to avoid canceling the function.
			// The function cannot be canceled even if the context is canceled
			// because multiple goroutines may depend on it.
			value = f(xcontext.Detach(ctx))

			// The function has completed. Update the value in the entry.
			e.mu.Lock()

			// Note: Because this defer will execute before the first defer,
			// we will hold the lock while we update the entry's wait channel.
			defer e.mu.Unlock()
			setref(e, value)
		}()
	}

	// Get a local copy of wait while we still hold the lock.
	wait := e.wait

	// Release the lock while we wait for the value.
	e.mu.Unlock()

	select {
	case <-wait:
		// We should now have a value. Lock the entry, and don't defer an unlock,
		// since we already have done so at the beginning of this function.
		e.mu.Lock()
		result := unref(e)

		// This keep alive makes sure that value is not garbage collected before
		// we call unref and acquire a strong reference to it.
		runtime.KeepAlive(value)
		return result, true
	case <-ctx.Done():
		// The context was canceled, but we have to lock the entry again,
		// since we already deferred an unlock at the beginning of this function.
		e.mu.Lock()
		return nil, false
	}
}

// setref is called to store a weak reference to a value into an entry.
// It assumes that the caller is holding the entry's lock.
func setref(e *entry, value interface{}) interface{} {
	// this is only called when the entry lock is already held
	data := (*[2]uintptr)(unsafe.Pointer(&value))
	// store the value back to the entry as a weak reference
	e.typ, e.ptr = data[0], data[1]
	e.ready = true
	if e.ptr != 0 {
		// Arrange to clear the weak reference when the object is garbage collected.
		runtime.SetFinalizer(value, func(_ interface{}) {
			e.mu.Lock()
			defer e.mu.Unlock()

			// Clear the now-invalid non-pointer.
			e.typ, e.ptr = 0, 0
			// The value is no longer available.
			e.ready = false
		})
	}
	return value
}

// unref returns a strong reference to value stored in the given entry.
// It assumes that the caller is holding the entry's lock.
func unref(e *entry) interface{} {
	// this is only called when the entry lock is already held
	var v interface{}
	data := (*[2]uintptr)(unsafe.Pointer(&v))

	// Note: This approach for computing weak references and converting between
	// weak and strong references would be rendered invalid if Go's runtime
	// changed to allow moving objects on the heap.
	// If such a change were to occur, some modifications would need to be made
	// to this library.
	data[0], data[1] = e.typ, e.ptr
	return v
}
