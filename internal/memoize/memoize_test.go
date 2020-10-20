// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memoize_test

import (
	"context"
	"strings"
	"testing"

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
	g1.Destroy()
	expectGet(t, h2, g2, "res")

	// With all generations destroyed, key should be re-evaluated.
	g2.Destroy()
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
	g2.Inherit(h1, h2)

	g1.Destroy()
	expectGet(t, h1, g2, &v1)
	expectGet(t, h2, g2, &v2)
	for k, v := range map[string]*bool{"key1": &v1, "key2": &v2} {
		if got, want := *v, false; got != want {
			t.Errorf("after destroying g1, bound value %q is cleaned up", k)
		}
	}
	g2.Destroy()
	if got, want := v1, false; got != want {
		t.Error("after destroying g2, v1 is cleaned up")
	}
	if got, want := v2, true; got != want {
		t.Error("after destroying g2, v2 is not cleaned up")
	}
}
