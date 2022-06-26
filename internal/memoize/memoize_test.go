// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memoize_test

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/internal/memoize"
)

func TestGet(t *testing.T) {
	s := &memoize.Store{}
	g := s.Generation("x")

	evaled := 0

	h := g.Bind("key", func(context.Context, memoize.Arg) interface{} {
		evaled++
		return "res"
	}, nil)
	expectGet(t, h, g, "res")
	expectGet(t, h, g, "res")
	if evaled != 1 {
		t.Errorf("got %v calls to function, wanted 1", evaled)
	}
}

func expectGet(t *testing.T, h *memoize.Handle, g *memoize.Generation, wantV interface{}) {
	t.Helper()
	gotV, gotErr := h.Get(context.Background(), g, nil)
	if gotV != wantV || gotErr != nil {
		t.Fatalf("Get() = %v, %v, wanted %v, nil", gotV, gotErr, wantV)
	}
}

func expectGetError(t *testing.T, h *memoize.Handle, g *memoize.Generation, substr string) {
	gotV, gotErr := h.Get(context.Background(), g, nil)
	if gotErr == nil || !strings.Contains(gotErr.Error(), substr) {
		t.Fatalf("Get() = %v, %v, wanted err %q", gotV, gotErr, substr)
	}
}

func TestGenerations(t *testing.T) {
	s := &memoize.Store{}
	// Evaluate key in g1.
	g1 := s.Generation("g1")
	h1 := g1.Bind("key", func(context.Context, memoize.Arg) interface{} { return "res" }, nil)
	expectGet(t, h1, g1, "res")

	// Get key in g2. It should inherit the value from g1.
	g2 := s.Generation("g2")
	h2 := g2.Bind("key", func(context.Context, memoize.Arg) interface{} {
		t.Fatal("h2 should not need evaluation")
		return "error"
	}, nil)
	expectGet(t, h2, g2, "res")

	// With g1 destroyed, g2 should still work.
	g1.Destroy("TestGenerations")
	expectGet(t, h2, g2, "res")

	// With all generations destroyed, key should be re-evaluated.
	g2.Destroy("TestGenerations")
	g3 := s.Generation("g3")
	h3 := g3.Bind("key", func(context.Context, memoize.Arg) interface{} { return "new res" }, nil)
	expectGet(t, h3, g3, "new res")
}

func TestCleanup(t *testing.T) {
	s := &memoize.Store{}
	g1 := s.Generation("g1")
	v1 := false
	v2 := false
	cleanup := func(v interface{}) {
		*(v.(*bool)) = true
	}
	h1 := g1.Bind("key1", func(context.Context, memoize.Arg) interface{} {
		return &v1
	}, nil)
	h2 := g1.Bind("key2", func(context.Context, memoize.Arg) interface{} {
		return &v2
	}, cleanup)
	expectGet(t, h1, g1, &v1)
	expectGet(t, h2, g1, &v2)
	g2 := s.Generation("g2")
	g2.Inherit(h1)
	g2.Inherit(h2)

	g1.Destroy("TestCleanup")
	expectGet(t, h1, g2, &v1)
	expectGet(t, h2, g2, &v2)
	for k, v := range map[string]*bool{"key1": &v1, "key2": &v2} {
		if got, want := *v, false; got != want {
			t.Errorf("after destroying g1, bound value %q is cleaned up", k)
		}
	}
	g2.Destroy("TestCleanup")
	if got, want := v1, false; got != want {
		t.Error("after destroying g2, v1 is cleaned up")
	}
	if got, want := v2, true; got != want {
		t.Error("after destroying g2, v2 is not cleaned up")
	}
}

func TestHandleRefCounting(t *testing.T) {
	s := &memoize.Store{}
	g1 := s.Generation("g1")
	v1 := false
	v2 := false
	h1, release1 := g1.GetHandle("key1", func(context.Context, memoize.Arg) interface{} {
		return &v1
	})
	h2, release2 := g1.GetHandle("key2", func(context.Context, memoize.Arg) interface{} {
		return &v2
	})
	expectGet(t, h1, g1, &v1)
	expectGet(t, h2, g1, &v2)

	g2 := s.Generation("g2")
	expectGet(t, h1, g2, &v1)
	g1.Destroy("by test")
	expectGet(t, h2, g2, &v2)

	h2Copy, release2Copy := g2.GetHandle("key2", func(context.Context, memoize.Arg) interface{} {
		return &v1
	})
	if h2 != h2Copy {
		t.Error("NewHandle returned a new value while old is not destroyed yet")
	}
	expectGet(t, h2Copy, g2, &v2)
	g2.Destroy("by test")

	release2()
	if got, want := v2, false; got != want {
		t.Errorf("after destroying first v2 ref, got %v, want %v", got, want)
	}
	release2Copy()
	if got, want := v1, false; got != want {
		t.Errorf("after destroying v2, got %v, want %v", got, want)
	}
	release1()

	g3 := s.Generation("g3")
	h2Copy, release2Copy = g3.GetHandle("key2", func(context.Context, memoize.Arg) interface{} {
		return &v2
	})
	if h2 == h2Copy {
		t.Error("NewHandle returned previously destroyed value")
	}
	release2Copy()
	g3.Destroy("by test")
}

func TestHandleDestroyedWhileRunning(t *testing.T) {
	// Test that calls to Handle.Get return even if the handle is destroyed while
	// running.

	s := &memoize.Store{}
	g := s.Generation("g")
	c := make(chan int)

	var v int
	h, release := g.GetHandle("key", func(ctx context.Context, _ memoize.Arg) interface{} {
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
		got, err = h.Get(ctx, g, nil)
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
