// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"sync"
	"sync/atomic"
)

// This file implements M8-S1 `logos/rt.Group` (SPEC Sec 2.6, T-206): a
// bound set of goroutines with context cancellation on the first child
// error or panic, a WaitGroup join, and a semaphore for `limit`.
//
// Ownership: every child is owned by its group; the group context
// propagates cancellation to all children, and each child reports its
// outcome to the group before exiting. Concurrency bound: at most
// `limit` children run at once when limit > 0; a further Go blocks
// until a token frees (backpressure, no queue). The join tolerates
// children that spawn during the run: the WaitGroup carries one body
// seed while the group statement runs, so a spawn concurrent with Wait
// always finds a positive counter.
//
// A Group is single-use: one body sequence of Go calls, then one Wait.
// Go after Wait and a second Wait are misuse and panic through the
// WaitGroup, the same as misusing sync.WaitGroup.

// groupIDs mints group IDs; wraparound reuse is harmless because IDs
// are compared only among live parked waiters (M8-S4).
var groupIDs atomic.Uint64

// Group is a bound set of goroutines. Build it with NewGroup, spawn
// children with Go, and join them with Wait.
type Group struct {
	ctx    context.Context
	cancel context.CancelFunc
	sem    chan struct{}
	wg     sync.WaitGroup
	mu     sync.Mutex
	first  error
	id     uint64
}

// PanicError is the error of a group child that panicked: Value is the
// recovered value and Stack is the stack of the panicking goroutine as
// runtime/debug.Stack reports it.
type PanicError struct {
	Value any
	Stack []byte
}

// Error formats the panic as `logos: group child panicked: <value>`,
// spelling the value the way fmt's %v does for nil, errors, Stringers,
// booleans, integers, floats, and strings, and as `(T) 0xADDR` for any
// other kind (see panicvalue.go).
func (e *PanicError) Error() string {
	return "logos: group child panicked: " + panicValueText(e.Value)
}

// NewGroup builds a group whose children share a context cancelled on
// the first child error or panic, and always at join. limit bounds how
// many children run at once; limit <= 0 leaves concurrency unbounded.
func NewGroup(ctx context.Context, limit int) (*Group, context.Context) {
	c, cancel := context.WithCancel(ctx)
	g := &Group{ctx: c, cancel: cancel, id: groupIDs.Add(1)}
	if limit > 0 {
		g.sem = make(chan struct{}, limit)
	}
	g.wg.Add(1)
	return g, c
}

// Go spawns f as a child. f runs with the group context; a non-nil
// return or a panic fails the group with the first outcome winning and
// cancels the siblings. A panic becomes *PanicError. The join counter
// is taken before the semaphore: a spawn blocked on the semaphore
// still holds it, so Wait cannot return before that child runs.
func (g *Group) Go(f func(context.Context) error) {
	g.wg.Add(1)
	if g.sem != nil {
		g.sem <- struct{}{}
	}
	go func() {
		if g.sem != nil {
			defer func() { <-g.sem }()
		}
		defer g.wg.Done()
		logosTagGroup(g.id)
		defer logosTagGroup(0)
		defer func() {
			if r := recover(); r != nil {
				g.fail(&PanicError{Value: r, Stack: stackTrace()})
			}
		}()
		if err := f(g.ctx); err != nil {
			g.fail(err)
		}
	}()
}

// fail records the first child outcome and cancels the group. A nil
// error records nothing.
func (g *Group) fail(err error) {
	if err == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.first == nil {
		g.first = err
		g.cancel()
		logosCancelGroup(g.id)
	}
}

// Wait joins every child, cancels the child context, and reports the
// first child error or panic, or nil when all children returned nil.
func (g *Group) Wait() error {
	g.wg.Done()
	g.wg.Wait()
	g.cancel()
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.first
}
