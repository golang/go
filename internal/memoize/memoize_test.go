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

	h, release := store.Handle("key", func(context.Context, interface{}) interface{} {
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

func expectGet(t *testing.T, h *memoize.Handle, wantV interface{}) {
	t.Helper()
	gotV, gotErr := h.Get(context.Background(), nil)
	if gotV != wantV || gotErr != nil {
		t.Fatalf("Get() = %v, %v, wanted %v, nil", gotV, gotErr, wantV)
	}
}

func TestHandleRefCounting(t *testing.T) {
	var store memoize.Store
	v1 := false
	v2 := false
	h1, release1 := store.Handle("key1", func(context.Context, interface{}) interface{} {
		return &v1
	})
	h2, release2 := store.Handle("key2", func(context.Context, interface{}) interface{} {
		return &v2
	})
	expectGet(t, h1, &v1)
	expectGet(t, h2, &v2)

	expectGet(t, h1, &v1)
	expectGet(t, h2, &v2)

	h2Copy, release2Copy := store.Handle("key2", func(context.Context, interface{}) interface{} {
		return &v1
	})
	if h2 != h2Copy {
		t.Error("NewHandle returned a new value while old is not destroyed yet")
	}
	expectGet(t, h2Copy, &v2)

	release2()
	if got, want := v2, false; got != want {
		t.Errorf("after destroying first v2 ref, got %v, want %v", got, want)
	}
	release2Copy()
	if got, want := v1, false; got != want {
		t.Errorf("after destroying v2, got %v, want %v", got, want)
	}
	release1()

	h2Copy, release2Copy = store.Handle("key2", func(context.Context, interface{}) interface{} {
		return &v2
	})
	if h2 == h2Copy {
		t.Error("NewHandle returned previously destroyed value")
	}
	release2Copy()
}

func TestHandleDestroyedWhileRunning(t *testing.T) {
	// Test that calls to Handle.Get return even if the handle is destroyed while running.

	var store memoize.Store
	c := make(chan int)

	var v int
	h, release := store.Handle("key", func(ctx context.Context, _ interface{}) interface{} {
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

	c <- 0    // send once to enter the handle function
	release() // release before the handle function returns
	c <- 0    // let the handle function proceed

	wg.Wait()

	if err != nil {
		t.Errorf("Get() failed: %v", err)
	}
	if got != &v {
		t.Errorf("Get() = %v, want %v", got, v)
	}
}
