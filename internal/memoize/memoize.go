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
	"reflect"
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

type state int

const (
	stateIdle = iota
	stateRunning
	stateCompleted
)

// Handle is returned from a store when a key is bound to a function.
// It is then used to access the results of that function.
//
// A Handle starts out in idle state, waiting for something to demand its
// evaluation. It then transitions into running state. While it's running,
// waiters tracks the number of Get calls waiting for a result, and the done
// channel is used to notify waiters of the next state transition. Once the
// evaluation finishes, value is set, state changes to completed, and done
// is closed, unblocking waiters. Alternatively, as Get calls are cancelled,
// they decrement waiters. If it drops to zero, the inner context is cancelled,
// computation is abandoned, and state resets to idle to start the process over
// again.
type Handle struct {
	store *Store
	key   interface{}

	mu    sync.Mutex
	state state
	// done is set in running state, and closed when exiting it.
	done chan struct{}
	// cancel is set in running state. It cancels computation.
	cancel context.CancelFunc
	// waiters is the number of Gets outstanding.
	waiters uint
	// the function that will be used to populate the value
	function Function
	// value is set in completed state.
	value interface{}
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

//go:nocheckptr
// nocheckptr because: https://github.com/golang/go/issues/35125#issuecomment-545671062
func (s *Store) get(key interface{}) *Handle {
	// this must be called with the store mutex already held
	e, found := s.entries[key]
	if !found {
		return nil
	}
	return (*Handle)(unsafe.Pointer(e))
}

// Stats returns the number of each type of value in the store.
func (s *Store) Stats() map[reflect.Type]int {
	s.mu.Lock()
	defer s.mu.Unlock()

	result := map[reflect.Type]int{}
	for k := range s.entries {
		result[reflect.TypeOf(k)]++
	}
	return result
}

// DebugOnlyIterate iterates through all live cache entries and calls f on them.
// It should only be used for debugging purposes.
func (s *Store) DebugOnlyIterate(f func(k, v interface{})) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for k, e := range s.entries {
		h := (*Handle)(unsafe.Pointer(e))
		var v interface{}
		h.mu.Lock()
		if h.state == stateCompleted {
			v = h.value
		}
		h.mu.Unlock()
		if v == nil {
			continue
		}
		f(k, v)
	}
}

// Cached returns the value associated with a handle.
//
// It will never cause the value to be generated.
// It will return the cached value, if present.
func (h *Handle) Cached() interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.state == stateCompleted {
		return h.value
	}
	return nil
}

// Get returns the value associated with a handle.
//
// If the value is not yet ready, the underlying function will be invoked.
// This activates the handle, and it will remember the value for as long as it exists.
// If ctx is cancelled, Get returns nil.
func (h *Handle) Get(ctx context.Context) (interface{}, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	h.mu.Lock()
	switch h.state {
	case stateIdle:
		return h.run(ctx)
	case stateRunning:
		return h.wait(ctx)
	case stateCompleted:
		defer h.mu.Unlock()
		return h.value, nil
	default:
		panic("unknown state")
	}
}

// run starts h.function and returns the result. h.mu must be locked.
func (h *Handle) run(ctx context.Context) (interface{}, error) {
	childCtx, cancel := context.WithCancel(xcontext.Detach(ctx))
	h.cancel = cancel
	h.state = stateRunning
	h.done = make(chan struct{})
	function := h.function // Read under the lock
	go func() {
		// Just in case the function does something expensive without checking
		// the context, double-check we're still alive.
		if childCtx.Err() != nil {
			return
		}
		v := function(childCtx)
		if childCtx.Err() != nil {
			return
		}

		h.mu.Lock()
		defer h.mu.Unlock()
		// It's theoretically possible that the handle has been cancelled out
		// of the run that started us, and then started running again since we
		// checked childCtx above. Even so, that should be harmless, since each
		// run should produce the same results.
		if h.state != stateRunning {
			return
		}
		h.value = v
		h.function = nil
		h.state = stateCompleted
		close(h.done)
	}()

	return h.wait(ctx)
}

// wait waits for the value to be computed, or ctx to be cancelled. h.mu must be locked.
func (h *Handle) wait(ctx context.Context) (interface{}, error) {
	h.waiters++
	done := h.done
	h.mu.Unlock()

	select {
	case <-done:
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.state == stateCompleted {
			return h.value, nil
		}
		return nil, nil
	case <-ctx.Done():
		h.mu.Lock()
		defer h.mu.Unlock()
		h.waiters--
		if h.waiters == 0 && h.state == stateRunning {
			h.cancel()
			close(h.done)
			h.state = stateIdle
			h.done = nil
			h.cancel = nil
		}
		return nil, ctx.Err()
	}
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
