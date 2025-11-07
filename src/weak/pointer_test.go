// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package weak_test

import (
	"context"
	"internal/goarch"
	"runtime"
	"sync"
	"testing"
	"time"
	"unsafe"
	"weak"
)

type T struct {
	// N.B. T is what it is to avoid having test values get tiny-allocated
	// in the same block as the weak handle, but since the fix to
	// go.dev/issue/76007, this should no longer be possible.
	// TODO(mknyszek): Consider using tiny-allocated values for all the tests.
	t *T
	a int
	b int
}

func TestPointer(t *testing.T) {
	var zero weak.Pointer[T]
	if zero.Value() != nil {
		t.Error("Value of zero value of weak.Pointer is not nil")
	}
	zeroNil := weak.Make[T](nil)
	if zeroNil.Value() != nil {
		t.Error("Value of weak.Make[T](nil) is not nil")
	}

	bt := new(T)
	wt := weak.Make(bt)
	if st := wt.Value(); st != bt {
		t.Fatalf("weak pointer is not the same as strong pointer: %p vs. %p", st, bt)
	}
	// bt is still referenced.
	runtime.GC()

	if st := wt.Value(); st != bt {
		t.Fatalf("weak pointer is not the same as strong pointer after GC: %p vs. %p", st, bt)
	}
	// bt is no longer referenced.
	runtime.GC()

	if st := wt.Value(); st != nil {
		t.Fatalf("expected weak pointer to be nil, got %p", st)
	}
}

func TestPointerEquality(t *testing.T) {
	var zero weak.Pointer[T]
	zeroNil := weak.Make[T](nil)
	if zero != zeroNil {
		t.Error("weak.Make[T](nil) != zero value of weak.Pointer[T]")
	}

	bt := make([]*T, 10)
	wt := make([]weak.Pointer[T], 10)
	wo := make([]weak.Pointer[int], 10)
	for i := range bt {
		bt[i] = new(T)
		wt[i] = weak.Make(bt[i])
		wo[i] = weak.Make(&bt[i].a)
	}
	for i := range bt {
		st := wt[i].Value()
		if st != bt[i] {
			t.Fatalf("weak pointer is not the same as strong pointer: %p vs. %p", st, bt[i])
		}
		if wp := weak.Make(st); wp != wt[i] {
			t.Fatalf("new weak pointer not equal to existing weak pointer: %v vs. %v", wp, wt[i])
		}
		if wp := weak.Make(&st.a); wp != wo[i] {
			t.Fatalf("new weak pointer not equal to existing weak pointer: %v vs. %v", wp, wo[i])
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
		st := wt[i].Value()
		if st != bt[i] {
			t.Fatalf("weak pointer is not the same as strong pointer: %p vs. %p", st, bt[i])
		}
		if wp := weak.Make(st); wp != wt[i] {
			t.Fatalf("new weak pointer not equal to existing weak pointer: %v vs. %v", wp, wt[i])
		}
		if wp := weak.Make(&st.a); wp != wo[i] {
			t.Fatalf("new weak pointer not equal to existing weak pointer: %v vs. %v", wp, wo[i])
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
		st := wt[i].Value()
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
		if wt.Value() != nil {
			t.Errorf("weak pointer did not go nil before finalizer ran")
		}
		done <- struct{}{}
	})

	// Make sure the weak pointer stays around while bt is live.
	runtime.GC()
	if wt.Value() == nil {
		t.Errorf("weak pointer went nil too soon")
	}
	runtime.KeepAlive(bt)

	// bt is no longer referenced.
	//
	// Run one cycle to queue the finalizer.
	runtime.GC()
	if wt.Value() != nil {
		t.Errorf("weak pointer did not go nil when finalizer was enqueued")
	}

	// Wait for the finalizer to run.
	<-done

	// The weak pointer should still be nil after the finalizer runs.
	runtime.GC()
	if wt.Value() != nil {
		t.Errorf("weak pointer is non-nil even after finalization: %v", wt)
	}
}

func TestPointerCleanup(t *testing.T) {
	bt := new(T)
	wt := weak.Make(bt)
	done := make(chan struct{}, 1)
	runtime.AddCleanup(bt, func(_ bool) {
		if wt.Value() != nil {
			t.Errorf("weak pointer did not go nil before cleanup was executed")
		}
		done <- struct{}{}
	}, true)

	// Make sure the weak pointer stays around while bt is live.
	runtime.GC()
	if wt.Value() == nil {
		t.Errorf("weak pointer went nil too soon")
	}
	runtime.KeepAlive(bt)

	// bt is no longer referenced.
	//
	// Run one cycle to queue the cleanup.
	runtime.GC()
	if wt.Value() != nil {
		t.Errorf("weak pointer did not go nil when cleanup was enqueued")
	}

	// Wait for the cleanup to run.
	<-done

	// The weak pointer should still be nil after the cleanup runs.
	runtime.GC()
	if wt.Value() != nil {
		t.Errorf("weak pointer is non-nil even after cleanup: %v", wt)
	}
}

func TestPointerSize(t *testing.T) {
	var p weak.Pointer[T]
	size := unsafe.Sizeof(p)
	if size != goarch.PtrSize {
		t.Errorf("weak.Pointer[T] size = %d, want %d", size, goarch.PtrSize)
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
	// 2. To call Value while the GC is in the mark phase,
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
					bt = wt.Value()
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

func TestIssue70739(t *testing.T) {
	x := make([]*int, 4<<16)
	wx1 := weak.Make(&x[1<<16])
	wx2 := weak.Make(&x[1<<16])
	if wx1 != wx2 {
		t.Fatal("failed to look up special and made duplicate weak handle; see issue #70739")
	}
}

var immortal T

func TestImmortalPointer(t *testing.T) {
	w0 := weak.Make(&immortal)
	if weak.Make(&immortal) != w0 {
		t.Error("immortal weak pointers to the same pointer not equal")
	}
	w0a := weak.Make(&immortal.a)
	w0b := weak.Make(&immortal.b)
	if w0a == w0b {
		t.Error("separate immortal pointers (same object) have the same pointer")
	}
	if got, want := w0.Value(), &immortal; got != want {
		t.Errorf("immortal weak pointer to %p has unexpected Value %p", want, got)
	}
	if got, want := w0a.Value(), &immortal.a; got != want {
		t.Errorf("immortal weak pointer to %p has unexpected Value %p", want, got)
	}
	if got, want := w0b.Value(), &immortal.b; got != want {
		t.Errorf("immortal weak pointer to %p has unexpected Value %p", want, got)
	}

	// Run a couple of cycles.
	runtime.GC()
	runtime.GC()

	// All immortal weak pointers should never get cleared.
	if got, want := w0.Value(), &immortal; got != want {
		t.Errorf("immortal weak pointer to %p has unexpected Value %p", want, got)
	}
	if got, want := w0a.Value(), &immortal.a; got != want {
		t.Errorf("immortal weak pointer to %p has unexpected Value %p", want, got)
	}
	if got, want := w0b.Value(), &immortal.b; got != want {
		t.Errorf("immortal weak pointer to %p has unexpected Value %p", want, got)
	}
}

func TestPointerTiny(t *testing.T) {
	runtime.GC() // Clear tiny-alloc caches.

	const N = 1000
	wps := make([]weak.Pointer[int], N)
	for i := range N {
		// N.B. *x is just an int, so the value is very likely
		// tiny-allocated alongside the weak handle, assuming bug
		// from go.dev/issue/76007 exists.
		x := new(int)
		*x = i
		wps[i] = weak.Make(x)
	}

	// Get the cleanups to start running.
	runtime.GC()

	// Expect at least 3/4ths of the weak pointers to have gone nil.
	//
	// Note that we provide some leeway since it's possible our allocation
	// gets grouped with some other long-lived tiny allocation, but this
	// shouldn't be the case for the vast majority of allocations.
	n := 0
	for _, wp := range wps {
		if wp.Value() == nil {
			n++
		}
	}
	const want = 3 * N / 4
	if n < want {
		t.Fatalf("not enough weak pointers are nil: expected at least %v, got %v", want, n)
	}
}
