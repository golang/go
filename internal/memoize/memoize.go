// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package memoize supports memoizing the return values of functions with
// idempotent results that are expensive to compute.
//
// To use this package, create a Store, call its Handle method to
// acquire a handle to (aka a "promise" of) the future result of a
// function, and call Handle.Get to obtain the result. Get may block
// if the function has not finished (or started).
package memoize

import (
	"context"
	"fmt"
	"reflect"
	"runtime/trace"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/internal/xcontext"
)

// Store binds keys to functions, returning handles that can be used to access
// the functions results.
type Store struct {
	handlesMu sync.Mutex // lock ordering: Store.handlesMu before Handle.mu
	handles   map[interface{}]*Handle
}

// A RefCounted is a value whose functional lifetime is determined by
// reference counting.
//
// Its Acquire method is called before the Function is invoked, and
// the corresponding release is called when the Function returns.
// Usually both events happen within a single call to Get, so Get
// would be fine with a "borrowed" reference, but if the context is
// cancelled, Get may return before the Function is complete, causing
// the argument to escape, and potential premature destruction of the
// value. For a reference-counted type, this requires a pair of
// increment/decrement operations to extend its life.
type RefCounted interface {
	// Acquire prevents the value from being destroyed until the
	// returned function is called.
	Acquire() func()
}

// Function is the type for functions that can be memoized.
//
// If the arg is a RefCounted, its Acquire/Release operations are called.
type Function func(ctx context.Context, arg interface{}) interface{}

type state int

// TODO(rfindley): consider removing stateIdle. Why create a handle if you
// aren't certain you're going to need its result? And if you know you need its
// result, why wait to begin computing it?
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
	key interface{}
	mu  sync.Mutex // lock ordering: Store.handlesMu before Handle.mu

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

	refcount int32 // accessed using atomic load/store
}

// Handle returns a reference-counted handle for the future result of
// calling the specified function. Calls to Handle with the same key
// return the same handle, and all calls to Handle.Get on a given
// handle return the same result but the function is called at most once.
//
// The caller must call the returned function to decrement the
// handle's reference count when it is no longer needed.
func (store *Store) Handle(key interface{}, function Function) (*Handle, func()) {
	if function == nil {
		panic("nil function")
	}

	store.handlesMu.Lock()
	h, ok := store.handles[key]
	if !ok {
		// new handle
		h = &Handle{
			key:      key,
			function: function,
			refcount: 1,
		}

		if store.handles == nil {
			store.handles = map[interface{}]*Handle{}
		}
		store.handles[key] = h
	} else {
		// existing handle
		atomic.AddInt32(&h.refcount, 1)
	}
	store.handlesMu.Unlock()

	release := func() {
		if atomic.AddInt32(&h.refcount, -1) == 0 {
			store.handlesMu.Lock()
			delete(store.handles, h.key)
			store.handlesMu.Unlock()
		}
	}
	return h, release
}

// Stats returns the number of each type of value in the store.
func (s *Store) Stats() map[reflect.Type]int {
	result := map[reflect.Type]int{}

	s.handlesMu.Lock()
	defer s.handlesMu.Unlock()

	for k := range s.handles {
		result[reflect.TypeOf(k)]++
	}
	return result
}

// DebugOnlyIterate iterates through all live cache entries and calls f on them.
// It should only be used for debugging purposes.
func (s *Store) DebugOnlyIterate(f func(k, v interface{})) {
	s.handlesMu.Lock()
	defer s.handlesMu.Unlock()

	for k, h := range s.handles {
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
// If ctx is cancelled, Get returns (nil, Canceled).
func (h *Handle) Get(ctx context.Context, arg interface{}) (interface{}, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	h.mu.Lock()
	switch h.state {
	case stateIdle:
		return h.run(ctx, arg)
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
func (h *Handle) run(ctx context.Context, arg interface{}) (interface{}, error) {
	childCtx, cancel := context.WithCancel(xcontext.Detach(ctx))
	h.cancel = cancel
	h.state = stateRunning
	h.done = make(chan struct{})
	function := h.function // Read under the lock

	// Make sure that the argument isn't destroyed while we're running in it.
	release := func() {}
	if rc, ok := arg.(RefCounted); ok {
		release = rc.Acquire()
	}

	go func() {
		trace.WithRegion(childCtx, fmt.Sprintf("Handle.run %T", h.key), func() {
			defer release()
			// Just in case the function does something expensive without checking
			// the context, double-check we're still alive.
			if childCtx.Err() != nil {
				return
			}
			v := function(childCtx, arg)
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
		})
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
