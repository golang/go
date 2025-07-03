// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package errgroup provides synchronization, error propagation, and Context
// cancelation for groups of goroutines working on subtasks of a common task.
//
// [errgroup.Group] is related to [sync.WaitGroup] but adds handling of tasks
// returning errors.
package errgroup

import (
	"context"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
)

type token struct{}

// A Group is a collection of goroutines working on subtasks that are part of
// the same overall task. A Group should not be reused for different tasks.
//
// A zero Group is valid, has no limit on the number of active goroutines,
// and does not cancel on error.
type Group struct {
	cancel func(error)

	wg sync.WaitGroup

	sem chan token

	errOnce sync.Once
	err     error

	mu         sync.Mutex
	panicValue any  // = PanicError | PanicValue; non-nil if some Group.Go goroutine panicked.
	abnormal   bool // some Group.Go goroutine terminated abnormally (panic or goexit).
}

func (g *Group) done() {
	if g.sem != nil {
		<-g.sem
	}
	g.wg.Done()
}

// WithContext returns a new Group and an associated Context derived from ctx.
//
// The derived Context is canceled the first time a function passed to Go
// returns a non-nil error or the first time Wait returns, whichever occurs
// first.
func WithContext(ctx context.Context) (*Group, context.Context) {
	ctx, cancel := context.WithCancelCause(ctx)
	return &Group{cancel: cancel}, ctx
}

// Wait blocks until all function calls from the Go method have returned
// normally, then returns the first non-nil error (if any) from them.
//
// If any of the calls panics, Wait panics with a [PanicValue];
// and if any of them calls [runtime.Goexit], Wait calls runtime.Goexit.
func (g *Group) Wait() error {
	g.wg.Wait()
	if g.cancel != nil {
		g.cancel(g.err)
	}
	if g.panicValue != nil {
		panic(g.panicValue)
	}
	if g.abnormal {
		runtime.Goexit()
	}
	return g.err
}

// Go calls the given function in a new goroutine.
//
// The first call to Go must happen before a Wait.
// It blocks until the new goroutine can be added without the number of
// goroutines in the group exceeding the configured limit.
//
// The first goroutine in the group that returns a non-nil error, panics, or
// invokes [runtime.Goexit] will cancel the associated Context, if any.
func (g *Group) Go(f func() error) {
	if g.sem != nil {
		g.sem <- token{}
	}

	g.add(f)
}

func (g *Group) add(f func() error) {
	g.wg.Add(1)
	go func() {
		defer g.done()
		normalReturn := false
		defer func() {
			if normalReturn {
				return
			}
			v := recover()
			g.mu.Lock()
			defer g.mu.Unlock()
			if !g.abnormal {
				if g.cancel != nil {
					g.cancel(g.err)
				}
				g.abnormal = true
			}
			if v != nil && g.panicValue == nil {
				switch v := v.(type) {
				case error:
					g.panicValue = PanicError{
						Recovered: v,
						Stack:     debug.Stack(),
					}
				default:
					g.panicValue = PanicValue{
						Recovered: v,
						Stack:     debug.Stack(),
					}
				}
			}
		}()

		err := f()
		normalReturn = true
		if err != nil {
			g.errOnce.Do(func() {
				g.err = err
				if g.cancel != nil {
					g.cancel(g.err)
				}
			})
		}
	}()
}

// TryGo calls the given function in a new goroutine only if the number of
// active goroutines in the group is currently below the configured limit.
//
// The return value reports whether the goroutine was started.
func (g *Group) TryGo(f func() error) bool {
	if g.sem != nil {
		select {
		case g.sem <- token{}:
			// Note: this allows barging iff channels in general allow barging.
		default:
			return false
		}
	}

	g.add(f)
	return true
}

// SetLimit limits the number of active goroutines in this group to at most n.
// A negative value indicates no limit.
// A limit of zero will prevent any new goroutines from being added.
//
// Any subsequent call to the Go method will block until it can add an active
// goroutine without exceeding the configured limit.
//
// The limit must not be modified while any goroutines in the group are active.
func (g *Group) SetLimit(n int) {
	if n < 0 {
		g.sem = nil
		return
	}
	if len(g.sem) != 0 {
		panic(fmt.Errorf("errgroup: modify limit while %v goroutines in the group are still active", len(g.sem)))
	}
	g.sem = make(chan token, n)
}

// PanicError wraps an error recovered from an unhandled panic
// when calling a function passed to Go or TryGo.
type PanicError struct {
	Recovered error
	Stack     []byte // result of call to [debug.Stack]
}

func (p PanicError) Error() string {
	if len(p.Stack) > 0 {
		return fmt.Sprintf("recovered from errgroup.Group: %v\n%s", p.Recovered, p.Stack)
	}
	return fmt.Sprintf("recovered from errgroup.Group: %v", p.Recovered)
}

func (p PanicError) Unwrap() error { return p.Recovered }

// PanicValue wraps a value that does not implement the error interface,
// recovered from an unhandled panic when calling a function passed to Go or
// TryGo.
type PanicValue struct {
	Recovered any
	Stack     []byte // result of call to [debug.Stack]
}

func (p PanicValue) String() string {
	if len(p.Stack) > 0 {
		return fmt.Sprintf("recovered from errgroup.Group: %v\n%s", p.Recovered, p.Stack)
	}
	return fmt.Sprintf("recovered from errgroup.Group: %v", p.Recovered)
}
