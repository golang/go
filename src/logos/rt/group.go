// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"sync"
)

// This file implements M8-S1 `logos/rt.Group` (SPEC Sec 2.6, T-206): a
// bound set of goroutines with context cancellation on the first child
// error or panic, a live-count join, and a semaphore for `limit`.
//
// Ownership: every child is owned by its group; the group context
// propagates cancellation to all children, and each child reports its
// outcome to the group before exiting. Concurrency bound: at most
// `limit` children run at once when limit > 0; a further Go blocks
// until a token frees (backpressure, no queue). The join tolerates
// children that spawn during the run: the live count carries one body
// seed while the group statement runs, so a spawn concurrent with Wait
// always finds a positive count.
//
// A Group is single-use: one body sequence of Go calls, then one Wait.
// Go after Wait panics, because the slot of the group has gone back to
// the runtime and a late child would carry it into another group. A
// second Wait is misuse and panics.

// Group is a bound set of goroutines. Build it with NewGroup, spawn
// children with Go, and join them with Wait.
type Group struct {
	ctx    context.Context
	cancel context.CancelFunc
	sem    chan struct{}
	mu     sync.Mutex
	cond   sync.Cond
	live   int
	waited bool
	joined bool
	first  error
	slot   uint32
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
// The group holds a runtime slot, the tag of its children for netpoll
// wake, until Wait has joined them all.
func NewGroup(ctx context.Context, limit int) (*Group, context.Context) {
	c, cancel := context.WithCancel(ctx)
	g := &Group{ctx: c, cancel: cancel, slot: logosGroupAcquire()}
	g.cond.L = &g.mu
	if limit > 0 {
		g.sem = make(chan struct{}, limit)
	}
	g.live = 1
	return g, c
}

// Go spawns f as a child. f runs with the group context; a non-nil
// return or a panic fails the group with the first outcome winning and
// cancels the siblings. A panic becomes *PanicError. The live count is
// taken before the semaphore: a spawn blocked on the semaphore still
// holds it, so Wait cannot return before that child runs. Go
// after Wait has returned panics with "logos: Group.Go called after
// Wait".
func (g *Group) Go(f func(context.Context) error) {
	g.mu.Lock()
	if g.joined {
		g.mu.Unlock()
		panic("logos: Group.Go called after Wait")
	}
	g.live++
	g.mu.Unlock()
	if g.sem != nil {
		g.sem <- struct{}{}
	}
	go func() {
		if g.sem != nil {
			defer func() { <-g.sem }()
		}
		defer g.childDone()
		logosTagGroup(g.slot)
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
		logosCancelGroup(g.slot)
	}
}

func (g *Group) childDone() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.live--
	if g.live == 0 {
		g.cond.Broadcast()
	}
}

// Wait joins every child, cancels the child context, and reports the
// first child error or panic, or nil when all children returned nil.
// Every child has cleared its tag when the join completes, so the slot
// goes back to the runtime only then. A second Wait panics with
// "logos: Group.Wait called twice".
func (g *Group) Wait() error {
	g.mu.Lock()
	if g.waited {
		g.mu.Unlock()
		panic("logos: Group.Wait called twice")
	}
	g.waited = true
	g.live--
	for g.live > 0 {
		g.cond.Wait()
	}
	g.joined = true
	g.mu.Unlock()
	logosGroupRelease(g.slot)
	g.cancel()
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.first
}
