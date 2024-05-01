// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package weak_test

import (
	"internal/weak"
	"runtime"
	"testing"
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
