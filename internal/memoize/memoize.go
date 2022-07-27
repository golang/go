// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package memoize defines a "promise" abstraction that enables
// memoization of the result of calling an expensive but idempotent
// function.
//
// Call p = NewPromise(f) to obtain a promise for the future result of
// calling f(), and call p.Get() to obtain that result. All calls to
// p.Get return the result of a single call of f().
// Get blocks if the function has not finished (or started).
//
// A Store is a map of arbitrary keys to promises. Use Store.Promise
// to create a promise in the store. All calls to Handle(k) return the
// same promise as long as it is in the store. These promises are
// reference-counted and must be explicitly released. Once the last
// reference is released, the promise is removed from the store.
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

// Function is the type of a function that can be memoized.
//
// If the arg is a RefCounted, its Acquire/Release operations are called.
//
// The argument must not materially affect the result of the function
// in ways that are not captured by the promise's key, since if
// Promise.Get is called twice concurrently, with the same (implicit)
// key but different arguments, the Function is called only once but
// its result must be suitable for both callers.
//
// The main purpose of the argument is to avoid the Function closure
// needing to retain large objects (in practice: the snapshot) in
// memory that can be supplied at call time by any caller.
type Function func(ctx context.Context, arg interface{}) interface{}

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

// A Promise represents the future result of a call to a function.
type Promise struct {
	debug string // for observability

	// refcount is the reference count in the containing Store, used by
	// Store.Promise. It is guarded by Store.promisesMu on the containing Store.
	refcount int32

	mu sync.Mutex

	// A Promise starts out IDLE, waiting for something to demand
	// its evaluation. It then transitions into RUNNING state.
	//
	// While RUNNING, waiters tracks the number of Get calls
	// waiting for a result, and the done channel is used to
	// notify waiters of the next state transition. Once
	// evaluation finishes, value is set, state changes to
	// COMPLETED, and done is closed, unblocking waiters.
	//
	// Alternatively, as Get calls are cancelled, they decrement
	// waiters. If it drops to zero, the inner context is
	// cancelled, computation is abandoned, and state resets to
	// IDLE to start the process over again.
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

// NewPromise returns a promise for the future result of calling the
// specified function.
//
// The debug string is used to classify promises in logs and metrics.
// It should be drawn from a small set.
func NewPromise(debug string, function Function) *Promise {
	if function == nil {
		panic("nil function")
	}
	return &Promise{
		debug:    debug,
		function: function,
	}
}

type state int

const (
	stateIdle      = iota // newly constructed, or last waiter was cancelled
	stateRunning          // start was called and not cancelled
	stateCompleted        // function call ran to completion
)

// Cached returns the value associated with a promise.
//
// It will never cause the value to be generated.
// It will return the cached value, if present.
func (p *Promise) Cached() interface{} {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.state == stateCompleted {
		return p.value
	}
	return nil
}

// Get returns the value associated with a promise.
//
// All calls to Promise.Get on a given promise return the
// same result but the function is called (to completion) at most once.
//
// If the value is not yet ready, the underlying function will be invoked.
//
// If ctx is cancelled, Get returns (nil, Canceled).
// If all concurrent calls to Get are cancelled, the context provided
// to the function is cancelled. A later call to Get may attempt to
// call the function again.
func (p *Promise) Get(ctx context.Context, arg interface{}) (interface{}, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	p.mu.Lock()
	switch p.state {
	case stateIdle:
		return p.run(ctx, arg)
	case stateRunning:
		return p.wait(ctx)
	case stateCompleted:
		defer p.mu.Unlock()
		return p.value, nil
	default:
		panic("unknown state")
	}
}

// run starts p.function and returns the result. p.mu must be locked.
func (p *Promise) run(ctx context.Context, arg interface{}) (interface{}, error) {
	childCtx, cancel := context.WithCancel(xcontext.Detach(ctx))
	p.cancel = cancel
	p.state = stateRunning
	p.done = make(chan struct{})
	function := p.function // Read under the lock

	// Make sure that the argument isn't destroyed while we're running in it.
	release := func() {}
	if rc, ok := arg.(RefCounted); ok {
		release = rc.Acquire()
	}

	go func() {
		trace.WithRegion(childCtx, fmt.Sprintf("Promise.run %s", p.debug), func() {
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

			p.mu.Lock()
			defer p.mu.Unlock()
			// It's theoretically possible that the promise has been cancelled out
			// of the run that started us, and then started running again since we
			// checked childCtx above. Even so, that should be harmless, since each
			// run should produce the same results.
			if p.state != stateRunning {
				return
			}

			p.value = v
			p.function = nil // aid GC
			p.state = stateCompleted
			close(p.done)
		})
	}()

	return p.wait(ctx)
}

// wait waits for the value to be computed, or ctx to be cancelled. p.mu must be locked.
func (p *Promise) wait(ctx context.Context) (interface{}, error) {
	p.waiters++
	done := p.done
	p.mu.Unlock()

	select {
	case <-done:
		p.mu.Lock()
		defer p.mu.Unlock()
		if p.state == stateCompleted {
			return p.value, nil
		}
		return nil, nil
	case <-ctx.Done():
		p.mu.Lock()
		defer p.mu.Unlock()
		p.waiters--
		if p.waiters == 0 && p.state == stateRunning {
			p.cancel()
			close(p.done)
			p.state = stateIdle
			p.done = nil
			p.cancel = nil
		}
		return nil, ctx.Err()
	}
}

// An EvictionPolicy controls the eviction behavior of keys in a Store when
// they no longer have any references.
type EvictionPolicy int

const (
	// ImmediatelyEvict evicts keys as soon as they no longer have references.
	ImmediatelyEvict EvictionPolicy = iota

	// NeverEvict does not evict keys.
	NeverEvict
)

// A Store maps arbitrary keys to reference-counted promises.
//
// The zero value is a valid Store, though a store may also be created via
// NewStore if a custom EvictionPolicy is required.
type Store struct {
	evictionPolicy EvictionPolicy

	promisesMu sync.Mutex
	promises   map[interface{}]*Promise
}

// NewStore creates a new store with the given eviction policy.
func NewStore(policy EvictionPolicy) *Store {
	return &Store{evictionPolicy: policy}
}

// Promise returns a reference-counted promise for the future result of
// calling the specified function.
//
// Calls to Promise with the same key return the same promise, incrementing its
// reference count.  The caller must call the returned function to decrement
// the promise's reference count when it is no longer needed. The returned
// function must not be called more than once.
//
// Once the last reference has been released, the promise is removed from the
// store.
func (store *Store) Promise(key interface{}, function Function) (*Promise, func()) {
	store.promisesMu.Lock()
	p, ok := store.promises[key]
	if !ok {
		p = NewPromise(reflect.TypeOf(key).String(), function)
		if store.promises == nil {
			store.promises = map[interface{}]*Promise{}
		}
		store.promises[key] = p
	}
	p.refcount++
	store.promisesMu.Unlock()

	var released int32
	release := func() {
		if !atomic.CompareAndSwapInt32(&released, 0, 1) {
			panic("release called more than once")
		}
		store.promisesMu.Lock()

		p.refcount--
		if p.refcount == 0 && store.evictionPolicy != NeverEvict {
			// Inv: if p.refcount > 0, then store.promises[key] == p.
			delete(store.promises, key)
		}
		store.promisesMu.Unlock()
	}

	return p, release
}

// Stats returns the number of each type of key in the store.
func (s *Store) Stats() map[reflect.Type]int {
	result := map[reflect.Type]int{}

	s.promisesMu.Lock()
	defer s.promisesMu.Unlock()

	for k := range s.promises {
		result[reflect.TypeOf(k)]++
	}
	return result
}

// DebugOnlyIterate iterates through the store and, for each completed
// promise, calls f(k, v) for the map key k and function result v.  It
// should only be used for debugging purposes.
func (s *Store) DebugOnlyIterate(f func(k, v interface{})) {
	s.promisesMu.Lock()
	defer s.promisesMu.Unlock()

	for k, p := range s.promises {
		if v := p.Cached(); v != nil {
			f(k, v)
		}
	}
}
