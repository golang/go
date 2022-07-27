// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memoize_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/internal/memoize"
)

func TestGet(t *testing.T) {
	var store memoize.Store

	evaled := 0

	h, release := store.Promise("key", func(context.Context, interface{}) interface{} {
		evaled++
		return "res"
	})
	defer release()
	expectGet(t, h, "res")
	expectGet(t, h, "res")
	if evaled != 1 {
		t.Errorf("got %v calls to function, wanted 1", evaled)
	}
}

func expectGet(t *testing.T, h *memoize.Promise, wantV interface{}) {
	t.Helper()
	gotV, gotErr := h.Get(context.Background(), nil)
	if gotV != wantV || gotErr != nil {
		t.Fatalf("Get() = %v, %v, wanted %v, nil", gotV, gotErr, wantV)
	}
}

func TestNewPromise(t *testing.T) {
	calls := 0
	f := func(context.Context, interface{}) interface{} {
		calls++
		return calls
	}

	// All calls to Get on the same promise return the same result.
	p1 := memoize.NewPromise("debug", f)
	expectGet(t, p1, 1)
	expectGet(t, p1, 1)

	// A new promise calls the function again.
	p2 := memoize.NewPromise("debug", f)
	expectGet(t, p2, 2)
	expectGet(t, p2, 2)

	// The original promise is unchanged.
	expectGet(t, p1, 1)
}

func TestStoredPromiseRefCounting(t *testing.T) {
	var store memoize.Store
	v1 := false
	v2 := false
	p1, release1 := store.Promise("key1", func(context.Context, interface{}) interface{} {
		return &v1
	})
	p2, release2 := store.Promise("key2", func(context.Context, interface{}) interface{} {
		return &v2
	})
	expectGet(t, p1, &v1)
	expectGet(t, p2, &v2)

	expectGet(t, p1, &v1)
	expectGet(t, p2, &v2)

	p2Copy, release2Copy := store.Promise("key2", func(context.Context, interface{}) interface{} {
		return &v1
	})
	if p2 != p2Copy {
		t.Error("Promise returned a new value while old is not destroyed yet")
	}
	expectGet(t, p2Copy, &v2)

	release2()
	if got, want := v2, false; got != want {
		t.Errorf("after destroying first v2 ref, got %v, want %v", got, want)
	}
	release2Copy()
	if got, want := v1, false; got != want {
		t.Errorf("after destroying v2, got %v, want %v", got, want)
	}
	release1()

	p2Copy, release2Copy = store.Promise("key2", func(context.Context, interface{}) interface{} {
		return &v2
	})
	if p2 == p2Copy {
		t.Error("Promise returned previously destroyed value")
	}
	release2Copy()
}

func TestPromiseDestroyedWhileRunning(t *testing.T) {
	// Test that calls to Promise.Get return even if the promise is destroyed while running.

	var store memoize.Store
	c := make(chan int)

	var v int
	h, release := store.Promise("key", func(ctx context.Context, _ interface{}) interface{} {
		<-c
		<-c
		if err := ctx.Err(); err != nil {
			t.Errorf("ctx.Err() = %v, want nil", err)
		}
		return &v
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second) // arbitrary timeout; may be removed if it causes flakes
	defer cancel()

	var wg sync.WaitGroup
	wg.Add(1)
	var got interface{}
	var err error
	go func() {
		got, err = h.Get(ctx, nil)
		wg.Done()
	}()

	c <- 0    // send once to enter the promise function
	release() // release before the promise function returns
	c <- 0    // let the promise function proceed

	wg.Wait()

	if err != nil {
		t.Errorf("Get() failed: %v", err)
	}
	if got != &v {
		t.Errorf("Get() = %v, want %v", got, v)
	}
}

func TestDoubleReleasePanics(t *testing.T) {
	var store memoize.Store
	_, release := store.Promise("key", func(ctx context.Context, _ interface{}) interface{} { return 0 })

	panicked := false

	func() {
		defer func() {
			if recover() != nil {
				panicked = true
			}
		}()
		release()
		release()
	}()

	if !panicked {
		t.Errorf("calling release() twice did not panic")
	}
}
