// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Outside callers racing Wait must join or fail cleanly: a Go that lands
// before the join completes runs to completion inside that join, and a
// Go after Wait returned panics with the documented message. The join
// never reports a WaitGroup reuse panic and never lets a child run
// after Wait returned.
//
// The racer keeps adding until Wait returns (a large backstop bounds
// the loop): a finite burst holds the join counter positive by
// construction, so only a racer still adding at the drain can land the
// forbidden post-drain Add. It yields every few attempts so its children and the
// waiter run on any GOMAXPROCS: without the yield the later-spawned
// waiter wins the runnext slot every iteration and the racer starves
// until the join is over, never exercising the race.
func TestGroupOutsideGoRacingWait(t *testing.T) {
	const iterations = 300
	const maxAttempts = 1 << 20
	var sawStarted atomic.Bool
	for i := range iterations {
		g, _ := NewGroup(t.Context(), 0)
		gg := g
		hold := make(chan struct{})
		gg.Go(func(ctx context.Context) error {
			<-hold
			return nil
		})
		ready := make(chan struct{})
		var waitStarted atomic.Bool
		var unexpected atomic.Value
		var late atomic.Bool
		var waitReturned atomic.Bool
		var racerDone sync.WaitGroup
		racerDone.Go(func() {
			defer func() {
				if r := recover(); r != nil {
					if msg, _ := r.(string); msg != "logos: Group.Go called after Wait" {
						unexpected.Store(msg)
					}
				}
			}()
			for j := 0; j < maxAttempts && !waitReturned.Load(); j++ {
				if j == 5 {
					close(ready)
				}
				if j%8 == 0 {
					runtime.Gosched()
				}
				gg.Go(func(ctx context.Context) error {
					if waitStarted.Load() {
						sawStarted.Store(true)
					}
					// Every child returns nil, so the group context is
					// cancelled only at the join: a child that starts
					// with a done context started after Wait returned.
					if ctx.Err() != nil {
						late.Store(true)
					}
					return nil
				})
			}
		})
		<-ready
		close(hold)
		waitStarted.Store(true)
		type waitOut struct {
			err error
			pan any
		}
		waitRes := make(chan waitOut, 1)
		go func() {
			defer func() {
				if r := recover(); r != nil {
					waitRes <- waitOut{pan: r}
				}
			}()
			waitRes <- waitOut{err: gg.Wait()}
		}()
		select {
		case out := <-waitRes:
			if out.pan != nil {
				t.Fatalf("iter %d: Wait panicked with %v", i, out.pan)
			}
			if out.err != nil {
				t.Fatalf("iter %d: Wait = %v, want nil", i, out.err)
			}
		case <-time.After(10 * time.Second):
			if v := unexpected.Load(); v != nil {
				t.Fatalf("iter %d: outside Go panicked with %q and wedged the join", i, v)
			}
			t.Fatalf("iter %d: Wait hung: outside Go wedged the join", i)
		}
		waitReturned.Store(true)
		racerDone.Wait()
		if v := unexpected.Load(); v != nil {
			t.Fatalf("iter %d: outside Go panicked with %q", i, v)
		}
		if late.Load() {
			t.Fatalf("iter %d: child ran after Wait returned", i)
		}
	}
	if !sawStarted.Load() {
		t.Fatal("no racer attempt overlapped Wait: test did not exercise the race")
	}
}

// Go after Wait panics with the documented message and starts no child.
func TestGroupGoAfterWaitMessage(t *testing.T) {
	g, _ := NewGroup(t.Context(), 0)
	if err := g.Wait(); err != nil {
		t.Fatalf("Wait = %v, want nil", err)
	}
	var ran atomic.Bool
	func() {
		defer func() {
			r := recover()
			if r != "logos: Group.Go called after Wait" {
				t.Fatalf("Go after Wait panicked with %v, want documented message", r)
			}
		}()
		g.Go(func(ctx context.Context) error {
			ran.Store(true)
			return nil
		})
	}()
	if ran.Load() {
		t.Fatal("child ran after Wait")
	}
}

// A second Wait is misuse and panics instead of joining a dead group.
func TestGroupSecondWaitPanics(t *testing.T) {
	g, _ := NewGroup(t.Context(), 0)
	if err := g.Wait(); err != nil {
		t.Fatalf("Wait = %v, want nil", err)
	}
	func() {
		defer func() {
			r := recover()
			if r != "logos: Group.Wait called twice" {
				t.Fatalf("second Wait panicked with %v, want documented message", r)
			}
		}()
		_ = g.Wait()
	}()
}
