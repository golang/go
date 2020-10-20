// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package memoize supports memoizing the return values of functions with
// idempotent results that are expensive to compute.
//
// To use this package, build a store and use it to acquire handles with the
// Bind method.
//
package memoize

import (
	"context"
	"flag"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/internal/xcontext"
)

var (
	panicOnDestroyed = flag.Bool("memoize_panic_on_destroyed", false,
		"Panic when a destroyed generation is read rather than returning an error. "+
			"Panicking may make it easier to debug lifetime errors, especially when "+
			"used with GOTRACEBACK=crash to see all running goroutines.")
)

// Store binds keys to functions, returning handles that can be used to access
// the functions results.
type Store struct {
	mu sync.Mutex
	// handles is the set of values stored.
	handles map[interface{}]*Handle

	// generations is the set of generations live in this store.
	generations map[*Generation]struct{}
}

// Generation creates a new Generation associated with s. Destroy must be
// called on the returned Generation once it is no longer in use. name is
// for debugging purposes only.
func (s *Store) Generation(name string) *Generation {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handles == nil {
		s.handles = map[interface{}]*Handle{}
		s.generations = map[*Generation]struct{}{}
	}
	g := &Generation{store: s, name: name}
	s.generations[g] = struct{}{}
	return g
}

// A Generation is a logical point in time of the cache life-cycle. Cache
// entries associated with a Generation will not be removed until the
// Generation is destroyed.
type Generation struct {
	// destroyed is 1 after the generation is destroyed. Atomic.
	destroyed uint32
	store     *Store
	name      string
	// wg tracks the reference count of this generation.
	wg sync.WaitGroup
}

// Destroy waits for all operations referencing g to complete, then removes
// all references to g from cache entries. Cache entries that no longer
// reference any non-destroyed generation are removed. Destroy must be called
// exactly once for each generation.
func (g *Generation) Destroy() {
	g.wg.Wait()
	atomic.StoreUint32(&g.destroyed, 1)
	g.store.mu.Lock()
	defer g.store.mu.Unlock()
	for k, e := range g.store.handles {
		e.mu.Lock()
		if _, ok := e.generations[g]; ok {
			delete(e.generations, g) // delete even if it's dead, in case of dangling references to the entry.
			if len(e.generations) == 0 {
				delete(g.store.handles, k)
				e.state = stateDestroyed
				if e.cleanup != nil && e.value != nil {
					e.cleanup(e.value)
				}
			}
		}
		e.mu.Unlock()
	}
	delete(g.store.generations, g)
}

// Acquire creates a new reference to g, and returns a func to release that
// reference.
func (g *Generation) Acquire(ctx context.Context) func() {
	destroyed := atomic.LoadUint32(&g.destroyed)
	if ctx.Err() != nil {
		return func() {}
	}
	if destroyed != 0 {
		panic("acquire on destroyed generation " + g.name)
	}
	g.wg.Add(1)
	return g.wg.Done
}

// Arg is a marker interface that can be embedded to indicate a type is
// intended for use as a Function argument.
type Arg interface{ memoizeArg() }

// Function is the type for functions that can be memoized.
// The result must be a pointer.
type Function func(ctx context.Context, arg Arg) interface{}

type state int

const (
	stateIdle = iota
	stateRunning
	stateCompleted
	stateDestroyed
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
	mu  sync.Mutex

	// generations is the set of generations in which this handle is valid.
	generations map[*Generation]struct{}

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
	// cleanup, if non-nil, is used to perform any necessary clean-up on values
	// produced by function.
	cleanup func(interface{})
}

// Bind returns a handle for the given key and function.
//
// Each call to bind will return the same handle if it is already bound. Bind
// will always return a valid handle, creating one if needed. Each key can
// only have one handle at any given time. The value will be held at least
// until the associated generation is destroyed. Bind does not cause the value
// to be generated.
//
// If cleanup is non-nil, it will be called on any non-nil values produced by
// function when they are no longer referenced.
func (g *Generation) Bind(key interface{}, function Function, cleanup func(interface{})) *Handle {
	// panic early if the function is nil
	// it would panic later anyway, but in a way that was much harder to debug
	if function == nil {
		panic("the function passed to bind must not be nil")
	}
	if atomic.LoadUint32(&g.destroyed) != 0 {
		panic("operation on destroyed generation " + g.name)
	}
	g.store.mu.Lock()
	defer g.store.mu.Unlock()
	h, ok := g.store.handles[key]
	if !ok {
		h := &Handle{
			key:         key,
			function:    function,
			generations: map[*Generation]struct{}{g: {}},
			cleanup:     cleanup,
		}
		g.store.handles[key] = h
		return h
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.generations[g]; !ok {
		h.generations[g] = struct{}{}
	}
	return h
}

// Stats returns the number of each type of value in the store.
func (s *Store) Stats() map[reflect.Type]int {
	s.mu.Lock()
	defer s.mu.Unlock()

	result := map[reflect.Type]int{}
	for k := range s.handles {
		result[reflect.TypeOf(k)]++
	}
	return result
}

// DebugOnlyIterate iterates through all live cache entries and calls f on them.
// It should only be used for debugging purposes.
func (s *Store) DebugOnlyIterate(f func(k, v interface{})) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for k, e := range s.handles {
		var v interface{}
		e.mu.Lock()
		if e.state == stateCompleted {
			v = e.value
		}
		e.mu.Unlock()
		if v == nil {
			continue
		}
		f(k, v)
	}
}

func (g *Generation) Inherit(hs ...*Handle) {
	for _, h := range hs {
		if atomic.LoadUint32(&g.destroyed) != 0 {
			panic("inherit on destroyed generation " + g.name)
		}

		h.mu.Lock()
		defer h.mu.Unlock()
		if h.state == stateDestroyed {
			panic(fmt.Sprintf("inheriting destroyed handle %#v (type %T) into generation %v", h.key, h.key, g.name))
		}
		h.generations[g] = struct{}{}
	}
}

// Cached returns the value associated with a handle.
//
// It will never cause the value to be generated.
// It will return the cached value, if present.
func (h *Handle) Cached(g *Generation) interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.generations[g]; !ok {
		return nil
	}
	if h.state == stateCompleted {
		return h.value
	}
	return nil
}

// Get returns the value associated with a handle.
//
// If the value is not yet ready, the underlying function will be invoked.
// If ctx is cancelled, Get returns nil.
func (h *Handle) Get(ctx context.Context, g *Generation, arg Arg) (interface{}, error) {
	release := g.Acquire(ctx)
	defer release()

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	h.mu.Lock()
	if _, ok := h.generations[g]; !ok {
		h.mu.Unlock()

		err := fmt.Errorf("reading key %#v: generation %v is not known", h.key, g.name)
		if *panicOnDestroyed && ctx.Err() != nil {
			panic(err)
		}
		return nil, err
	}
	switch h.state {
	case stateIdle:
		return h.run(ctx, g, arg)
	case stateRunning:
		return h.wait(ctx)
	case stateCompleted:
		defer h.mu.Unlock()
		return h.value, nil
	case stateDestroyed:
		h.mu.Unlock()
		err := fmt.Errorf("Get on destroyed entry %#v (type %T) in generation %v", h.key, h.key, g.name)
		if *panicOnDestroyed {
			panic(err)
		}
		return nil, err
	default:
		panic("unknown state")
	}
}

// run starts h.function and returns the result. h.mu must be locked.
func (h *Handle) run(ctx context.Context, g *Generation, arg Arg) (interface{}, error) {
	childCtx, cancel := context.WithCancel(xcontext.Detach(ctx))
	h.cancel = cancel
	h.state = stateRunning
	h.done = make(chan struct{})
	function := h.function // Read under the lock

	// Make sure that the generation isn't destroyed while we're running in it.
	release := g.Acquire(ctx)
	go func() {
		defer release()
		// Just in case the function does something expensive without checking
		// the context, double-check we're still alive.
		if childCtx.Err() != nil {
			return
		}
		v := function(childCtx, arg)
		if childCtx.Err() != nil {
			// It's possible that v was computed despite the context cancellation. In
			// this case we should ensure that it is cleaned up.
			if h.cleanup != nil && v != nil {
				h.cleanup(v)
			}
			return
		}

		h.mu.Lock()
		defer h.mu.Unlock()
		// It's theoretically possible that the handle has been cancelled out
		// of the run that started us, and then started running again since we
		// checked childCtx above. Even so, that should be harmless, since each
		// run should produce the same results.
		if h.state != stateRunning {
			// v will never be used, so ensure that it is cleaned up.
			if h.cleanup != nil && v != nil {
				h.cleanup(v)
			}
			return
		}
		// At this point v will be cleaned up whenever h is destroyed.
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
