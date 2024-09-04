// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package weak_test

import (
	"context"
	"internal/weak"
	"runtime"
	"sync"
	"testing"
	"time"
)

type T struct {
	// N.B. This must contain a pointer, otherwise the weak handle might get placed
	// in a tiny block making the tests in this package flaky.
	t *T
	a int
}

func TestPointer(t *testing.T) {
	bt := new(T)
	wt := weak.Make(bt)
	if st := wt.Strong(); st != bt {
		t.Fatalf("weak pointer is not the same as strong pointer: %p vs. %p", st, bt)
	}
	// bt is still referenced.
	runtime.GC()

	if st := wt.Strong(); st != bt {
		t.Fatalf("weak pointer is not the same as strong pointer after GC: %p vs. %p", st, bt)
	}
	// bt is no longer referenced.
	runtime.GC()

	if st := wt.Strong(); st != nil {
		t.Fatalf("expected weak pointer to be nil, got %p", st)
	}
}

func TestPointerEquality(t *testing.T) {
	bt := make([]*T, 10)
	wt := make([]weak.Pointer[T], 10)
	for i := range bt {
		bt[i] = new(T)
		wt[i] = weak.Make(bt[i])
	}
	for i := range bt {
		st := wt[i].Strong()
		if st != bt[i] {
			t.Fatalf("weak pointer is not the same as strong pointer: %p vs. %p", st, bt[i])
		}
		if wp := weak.Make(st); wp != wt[i] {
			t.Fatalf("new weak pointer not equal to existing weak pointer: %v vs. %v", wp, wt[i])
		}
		if i == 0 {
			continue
		}
		if wt[i] == wt[i-1] {
			t.Fatalf("expected weak pointers to not be equal to each other, but got %v", wt[i])
		}
	}
	// bt is still referenced.
	runtime.GC()
	for i := range bt {
		st := wt[i].Strong()
		if st != bt[i] {
			t.Fatalf("weak pointer is not the same as strong pointer: %p vs. %p", st, bt[i])
		}
		if wp := weak.Make(st); wp != wt[i] {
			t.Fatalf("new weak pointer not equal to existing weak pointer: %v vs. %v", wp, wt[i])
		}
		if i == 0 {
			continue
		}
		if wt[i] == wt[i-1] {
			t.Fatalf("expected weak pointers to not be equal to each other, but got %v", wt[i])
		}
	}
	bt = nil
	// bt is no longer referenced.
	runtime.GC()
	for i := range bt {
		st := wt[i].Strong()
		if st != nil {
			t.Fatalf("expected weak pointer to be nil, got %p", st)
		}
		if i == 0 {
			continue
		}
		if wt[i] == wt[i-1] {
			t.Fatalf("expected weak pointers to not be equal to each other, but got %v", wt[i])
		}
	}
}

func TestPointerFinalizer(t *testing.T) {
	bt := new(T)
	wt := weak.Make(bt)
	done := make(chan struct{}, 1)
	runtime.SetFinalizer(bt, func(bt *T) {
		if wt.Strong() != nil {
			t.Errorf("weak pointer did not go nil before finalizer ran")
		}
		done <- struct{}{}
	})

	// Make sure the weak pointer stays around while bt is live.
	runtime.GC()
	if wt.Strong() == nil {
		t.Errorf("weak pointer went nil too soon")
	}
	runtime.KeepAlive(bt)

	// bt is no longer referenced.
	//
	// Run one cycle to queue the finalizer.
	runtime.GC()
	if wt.Strong() != nil {
		t.Errorf("weak pointer did not go nil when finalizer was enqueued")
	}

	// Wait for the finalizer to run.
	<-done

	// The weak pointer should still be nil after the finalizer runs.
	runtime.GC()
	if wt.Strong() != nil {
		t.Errorf("weak pointer is non-nil even after finalization: %v", wt)
	}
}

// Regression test for issue 69210.
//
// Weak-to-strong conversions must shade the new strong pointer, otherwise
// that might be creating the only strong pointer to a white object which
// is hidden in a blackened stack.
//
// Never fails if correct, fails with some high probability if incorrect.
func TestIssue69210(t *testing.T) {
	if testing.Short() {
		t.Skip("this is a stress test that takes seconds to run on its own")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	// What we're trying to do is manufacture the conditions under which this
	// bug happens. Specifically, we want:
	//
	// 1. To create a whole bunch of objects that are only weakly-pointed-to,
	// 2. To call Strong while the GC is in the mark phase,
	// 3. The new strong pointer to be missed by the GC,
	// 4. The following GC cycle to mark a free object.
	//
	// Unfortunately, (2) and (3) are hard to control, but we can increase
	// the likelihood by having several goroutines do (1) at once while
	// another goroutine constantly keeps us in the GC with runtime.GC.
	// Like throwing darts at a dart board until they land just right.
	// We can increase the likelihood of (4) by adding some delay after
	// creating the strong pointer, but only if it's non-nil. If it's nil,
	// that means it was already collected in which case there's no chance
	// of triggering the bug, so we want to retry as fast as possible.
	// Our heap here is tiny, so the GCs will go by fast.
	//
	// As of 2024-09-03, removing the line that shades pointers during
	// the weak-to-strong conversion causes this test to fail about 50%
	// of the time.

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			runtime.GC()

			select {
			case <-ctx.Done():
				return
			default:
			}
		}
	}()
	for range max(runtime.GOMAXPROCS(-1)-1, 1) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				for range 5 {
					bt := new(T)
					wt := weak.Make(bt)
					bt = nil
					time.Sleep(1 * time.Millisecond)
					bt = wt.Strong()
					if bt != nil {
						time.Sleep(4 * time.Millisecond)
						bt.t = bt
						bt.a = 12
					}
					runtime.KeepAlive(bt)
				}
				select {
				case <-ctx.Done():
					return
				default:
				}
			}
		}()
	}
	wg.Wait()
}
