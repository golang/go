// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package memoize supports memoizing the return values of functions with
// idempotent results that are expensive to compute.
//
// The memoized result is returned again the next time the function is invoked.
// To prevent excessive memory use, the return values are only remembered
// for as long as they still have a user.
//
// To use this package, build a store and use it to acquire handles with the
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
	entries map[interface{}]uintptr
}

// Function is the type for functions that can be memoized.
// The result must be a pointer.
type Function func(ctx context.Context) interface{}

// Handle is returned from a store when a key is bound to a function.
// It is then used to access the results of that function.
type Handle struct {
	mu    sync.Mutex
	store *Store
	key   interface{}
	// the function that will be used to populate the value
	function Function
	// the lazily poplulated value
	value interface{}
	// wait is used to block until the value is ready
	// will only be non nil if the generator is already running
	wait chan interface{}
	// the cancel function for the context being used by the generator
	// it can be used to abort the generator if the handle is garbage
	// collected.
	cancel context.CancelFunc
}

// Has returns true if they key is currently valid for this store.
func (s *Store) Has(key interface{}) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, found := s.entries[key]
	return found
}

// Bind returns a handle for the given key and function.
//
// Each call to bind will return the same handle if it is already bound.
// Bind will always return a valid handle, creating one if needed.
// Each key can only have one handle at any given time.
// The value will be held for as long as the handle is, once it has been
// generated.
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
	h := s.get(key)
	if h != nil {
		// we have a handle already, just return it
		return h
	}
	// we have not seen this key before, add a new entry
	if s.entries == nil {
		s.entries = make(map[interface{}]uintptr)
	}
	h = &Handle{
		store:    s,
		key:      key,
		function: function,
	}
	// now add the weak reference to the handle into the map
	s.entries[key] = uintptr(unsafe.Pointer(h))
	// add the deletion the entry when the handle is garbage collected
	runtime.SetFinalizer(h, release)
	return h
}

// Find returns the handle associated with a key, if it is bound.
//
// It cannot cause a new handle to be generated, and thus may return nil.
func (s *Store) Find(key interface{}) *Handle {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.get(key)
}

// Cached returns the value associated with a key.
//
// It cannot cause the value to be generated.
// It will return the cached value, if present.
func (s *Store) Cached(key interface{}) interface{} {
	h := s.Find(key)
	if h == nil {
		return nil
	}
	return h.Cached()
}

func (s *Store) get(key interface{}) *Handle {
	// this must be called with the store mutex already held
	e, found := s.entries[key]
	if !found {
		return nil
	}
	return (*Handle)(unsafe.Pointer(e))
}

// Cached returns the value associated with a handle.
//
// It will never cause the value to be generated.
// It will return the cached value, if present.
func (h *Handle) Cached() interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.value
}

// Get returns the value associated with a handle.
//
// If the value is not yet ready, the underlying function will be invoked.
// This activates the handle, and it will remember the value for as long as it exists.
func (h *Handle) Get(ctx context.Context) interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.function == nil {
		return h.value
	}
	// value not ready yet
	select {
	case h.value = <-h.run(ctx):
		// successfully got the value
		h.function = nil
		h.cancel = nil
		return h.value
	case <-ctx.Done():
		// cancelled outer context, leave the generator running
		// for someone else to pick up later
		return nil
	}
}

// run starts the generator if necessary and returns the value channel.
func (h *Handle) run(ctx context.Context) chan interface{} {
	if h.wait != nil {
		// generator already running
		return h.wait
	}
	// we use a length one "postbox" so the go routine can quit even if
	// nobody wants the result yet
	h.wait = make(chan interface{}, 1)
	ctx, cancel := context.WithCancel(xcontext.Detach(ctx))
	h.cancel = cancel
	go func() {
		// in here the handle lock is not held
		// we post the value back to the first caller that waits for it
		h.wait <- h.function(ctx)
		close(h.wait)
	}()
	return h.wait
}

func release(p interface{}) {
	h := p.(*Handle)
	h.store.mu.Lock()
	defer h.store.mu.Unlock()
	// there is a small gap between the garbage collector deciding that the handle
	// is liable for collection and the finalizer being called
	// if the handle is recovered during that time, you will end up with a valid
	// handle that no longer has an entry in the map, and that no longer has a
	// finalizer associated with it, but that is okay.
	delete(h.store.entries, h.key)
}
