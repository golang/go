// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package memoize supports memoizing the return values of functions with
// idempotent results that are expensive to compute.
//
// To use this package, build a store and use it to acquire handles with the
// Bind method.
package memoize

import (
	"context"
	"flag"
	"fmt"
	"reflect"
	"runtime/trace"
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
	handlesMu sync.Mutex // lock ordering: Store.handlesMu before Handle.mu
	handles   map[interface{}]*Handle
}

// Generation creates a new Generation associated with s. Destroy must be
// called on the returned Generation once it is no longer in use. name is
// for debugging purposes only.
func (s *Store) Generation(name string) *Generation {
	return &Generation{store: s, name: name}
}

// A Generation is a logical point in time of the cache life-cycle. Cache
// entries associated with a Generation will not be removed until the
// Generation is destroyed.
type Generation struct {
	// destroyed is 1 after the generation is destroyed. Atomic.
	destroyed uint32
	store     *Store
	name      string
	// destroyedBy describes the caller that togged destroyed from 0 to 1.
	destroyedBy string
	// wg tracks the reference count of this generation.
	wg sync.WaitGroup
}

// Destroy waits for all operations referencing g to complete, then removes
// all references to g from cache entries. Cache entries that no longer
// reference any non-destroyed generation are removed. Destroy must be called
// exactly once for each generation, and destroyedBy describes the caller.
func (g *Generation) Destroy(destroyedBy string) {
	g.wg.Wait()

	prevDestroyedBy := g.destroyedBy
	g.destroyedBy = destroyedBy
	if ok := atomic.CompareAndSwapUint32(&g.destroyed, 0, 1); !ok {
		panic("Destroy on generation " + g.name + " already destroyed by " + prevDestroyedBy)
	}

	g.store.handlesMu.Lock()
	defer g.store.handlesMu.Unlock()
	for _, h := range g.store.handles {
		if !h.trackGenerations {
			continue
		}
		h.mu.Lock()
		if _, ok := h.generations[g]; ok {
			delete(h.generations, g) // delete even if it's dead, in case of dangling references to the entry.
			if len(h.generations) == 0 {
				h.destroy(g.store)
			}
		}
		h.mu.Unlock()
	}
}

// Acquire creates a new reference to g, and returns a func to release that
// reference.
func (g *Generation) Acquire() func() {
	destroyed := atomic.LoadUint32(&g.destroyed)
	if destroyed != 0 {
		panic("acquire on generation " + g.name + " destroyed by " + g.destroyedBy)
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

// TODO(rfindley): remove stateDestroyed; Handles should not need to know
// whether or not they have been destroyed.
//
// TODO(rfindley): also consider removing stateIdle. Why create a handle if you
// aren't certain you're going to need its result? And if you know you need its
// result, why wait to begin computing it?
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
//
// Handles may be tracked by generations, or directly reference counted, as
// determined by the trackGenerations field. See the field comments for more
// information about the differences between these two forms.
//
// TODO(rfindley): eliminate generational handles.
type Handle struct {
	key interface{}
	mu  sync.Mutex // lock ordering: Store.handlesMu before Handle.mu

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
	//
	// cleanup is never set for reference counted handles.
	//
	// TODO(rfindley): remove this field once workspace folders no longer need to
	// be tracked.
	cleanup func(interface{})

	// If trackGenerations is set, this handle tracks generations in which it
	// is valid, via the generations field. Otherwise, it is explicitly reference
	// counted via the refCounter field.
	trackGenerations bool
	refCounter       int32
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
//
// It is responsibility of the caller to call Inherit on the handler whenever
// it should still be accessible by a next generation.
func (g *Generation) Bind(key interface{}, function Function, cleanup func(interface{})) *Handle {
	return g.getHandle(key, function, cleanup, true)
}

// GetHandle returns a handle for the given key and function with similar
// properties and behavior as Bind.
//
// As in opposite to Bind it returns a release callback which has to be called
// once this reference to handle is not needed anymore.
func (g *Generation) GetHandle(key interface{}, function Function) (*Handle, func()) {
	h := g.getHandle(key, function, nil, false)
	store := g.store
	release := func() {
		// Acquire store.handlesMu before mutating refCounter
		store.handlesMu.Lock()
		defer store.handlesMu.Unlock()

		h.mu.Lock()
		defer h.mu.Unlock()

		h.refCounter--
		if h.refCounter == 0 {
			// Don't call h.destroy: for reference counted handles we can't know when
			// they are no longer reachable from runnable goroutines. For example,
			// gopls could have a current operation that is using a packageHandle.
			// Destroying the handle here would cause that operation to hang.
			delete(store.handles, h.key)
		}
	}
	return h, release
}

func (g *Generation) getHandle(key interface{}, function Function, cleanup func(interface{}), trackGenerations bool) *Handle {
	// panic early if the function is nil
	// it would panic later anyway, but in a way that was much harder to debug
	if function == nil {
		panic("the function passed to bind must not be nil")
	}
	if atomic.LoadUint32(&g.destroyed) != 0 {
		panic("operation on generation " + g.name + " destroyed by " + g.destroyedBy)
	}
	g.store.handlesMu.Lock()
	defer g.store.handlesMu.Unlock()
	h, ok := g.store.handles[key]
	if !ok {
		h = &Handle{
			key:              key,
			function:         function,
			cleanup:          cleanup,
			trackGenerations: trackGenerations,
		}
		if trackGenerations {
			h.generations = make(map[*Generation]struct{}, 1)
		}

		if g.store.handles == nil {
			g.store.handles = map[interface{}]*Handle{}
		}
		g.store.handles[key] = h
	}

	h.incrementRef(g)
	return h
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

// Inherit makes h valid in generation g. It is concurrency-safe.
func (g *Generation) Inherit(h *Handle) {
	if atomic.LoadUint32(&g.destroyed) != 0 {
		panic("inherit on generation " + g.name + " destroyed by " + g.destroyedBy)
	}
	if !h.trackGenerations {
		panic("called Inherit on handle not created by Generation.Bind")
	}

	h.incrementRef(g)
}

// destroy marks h as destroyed. h.mu and store.handlesMu must be held.
func (h *Handle) destroy(store *Store) {
	h.state = stateDestroyed
	if h.cleanup != nil && h.value != nil {
		h.cleanup(h.value)
	}
	delete(store.handles, h.key)
}

func (h *Handle) incrementRef(g *Generation) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.state == stateDestroyed {
		panic(fmt.Sprintf("inheriting destroyed handle %#v (type %T) into generation %v", h.key, h.key, g.name))
	}

	if h.trackGenerations {
		h.generations[g] = struct{}{}
	} else {
		h.refCounter++
	}
}

// hasRefLocked reports whether h is valid in generation g. h.mu must be held.
func (h *Handle) hasRefLocked(g *Generation) bool {
	if !h.trackGenerations {
		return true
	}

	_, ok := h.generations[g]
	return ok
}

// Cached returns the value associated with a handle.
//
// It will never cause the value to be generated.
// It will return the cached value, if present.
func (h *Handle) Cached(g *Generation) interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.hasRefLocked(g) {
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
	release := g.Acquire()
	defer release()

	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	h.mu.Lock()
	if !h.hasRefLocked(g) {
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
	release := g.Acquire()
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

			if h.cleanup != nil && h.value != nil {
				// Clean up before overwriting an existing value.
				h.cleanup(h.value)
			}

			// At this point v will be cleaned up whenever h is destroyed.
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
