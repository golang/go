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
	})
	expectGet(t, h, g, "res")
	expectGet(t, h, g, "res")
	if evaled != 1 {
		t.Errorf("got %v calls to function, wanted 1", evaled)
	}
}

func expectGet(t *testing.T, h *memoize.Handle, g *memoize.Generation, wantV interface{}) {
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
	h1 := g1.Bind("key", func(context.Context, memoize.Arg) interface{} { return "res" })
	expectGet(t, h1, g1, "res")

	// Get key in g2. It should inherit the value from g1.
	g2 := s.Generation("g2")
	h2 := g2.Bind("key", func(context.Context, memoize.Arg) interface{} {
		t.Fatal("h2 should not need evaluation")
		return "error"
	})
	expectGet(t, h2, g2, "res")

	// With g1 destroyed, g2 should still work.
	g1.Destroy()
	expectGet(t, h2, g2, "res")

	// With all generations destroyed, key should be re-evaluated.
	g2.Destroy()
	g3 := s.Generation("g3")
	h3 := g3.Bind("key", func(context.Context, memoize.Arg) interface{} { return "new res" })
	expectGet(t, h3, g3, "new res")
}
