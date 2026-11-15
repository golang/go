// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arena_test

import (
	"fmt"
	"logos/arena"
	"runtime"
	"slices"
	"testing"
)

func regionPanic(t *testing.T, f func()) string {
	t.Helper()
	var msg string
	func() {
		defer func() {
			p := recover()
			if p == nil {
				return
			}
			if _, ok := p.(runtime.Error); !ok {
				t.Errorf("panic %v does not implement runtime.Error", p)
			}
			msg = fmt.Sprint(p)
		}()
		f()
	}()
	if msg == "" {
		t.Fatalf("expected a runtime panic, got none")
	}
	return msg
}

func TestArenaHeapSemantics(t *testing.T) {
	t.Run("open", func(t *testing.T) {
		if got := arena.Open(); got != (arena.Region{}) {
			t.Errorf("Open() = %v, want zero Region", got)
		}
	})

	t.Run("new", func(t *testing.T) {
		r := arena.Open()
		p1, p2 := arena.New[int](r), arena.New[int](r)
		if p1 == nil || p2 == nil {
			t.Fatalf("New returned nil: %v %v", p1, p2)
		}
		if p1 == p2 {
			t.Fatalf("two New calls share one pointer %p", p1)
		}
		if *p1 != 0 || *p2 != 0 {
			t.Errorf("*p1 = %d, *p2 = %d, want zeroed", *p1, *p2)
		}
	})

	t.Run("make", func(t *testing.T) {
		r := arena.Open()
		s := arena.Make[int](r, 5)
		if len(s) != 5 || cap(s) != 5 {
			t.Fatalf("Make(r, 5) has len %d cap %d, want 5 5", len(s), cap(s))
		}
		for i, v := range s {
			if v != 0 {
				t.Errorf("s[%d] = %d, want 0", i, v)
			}
		}
	})

	t.Run("make-negative-panic", func(t *testing.T) {
		r := arena.Open()
		n := -1
		heapMsg := regionPanic(t, func() { _ = make([]int, n) })
		arenaMsg := regionPanic(t, func() { _ = arena.Make[int](r, n) })
		if arenaMsg != heapMsg {
			t.Errorf("Make panic %q, want heap make panic %q", arenaMsg, heapMsg)
		}
	})

	t.Run("append-nil", func(t *testing.T) {
		r := arena.Open()
		got := arena.Append(r, []int(nil), 1)
		want := append([]int(nil), 1)
		if !slices.Equal(got, want) {
			t.Errorf("Append(r, nil, 1) = %v, want %v", got, want)
		}
		if cap(got) != cap(want) {
			t.Errorf("Append(r, nil, 1) cap %d, want %d", cap(got), cap(want))
		}
	})

	t.Run("append-spare", func(t *testing.T) {
		r := arena.Open()
		base := make([]int, 2, 8)
		s := base[:2]
		got := arena.Append(r, s, 9)
		if &got[0] != &base[0] {
			t.Errorf("Append with spare capacity moved the backing array")
		}
		want := append(s, 9)
		if !slices.Equal(got, want) {
			t.Errorf("Append(r, s, 9) = %v, want %v", got, want)
		}
		if cap(got) != cap(want) {
			t.Errorf("Append(r, s, 9) cap %d, want %d", cap(got), cap(want))
		}
	})

	t.Run("append-grow", func(t *testing.T) {
		r := arena.Open()
		s := []int{1, 2}
		full := s[:2:2]
		got := arena.Append(r, full, 3, 4)
		if &got[0] == &full[0] {
			t.Fatalf("Append past capacity kept the old backing array")
		}
		cp := make([]int, len(full), cap(full))
		copy(cp, full)
		want := append(cp, 3, 4)
		if !slices.Equal(got, want) {
			t.Errorf("Append(r, full, 3, 4) = %v, want %v", got, want)
		}
		if cap(got) != cap(want) {
			t.Errorf("Append(r, full, 3, 4) cap %d, want heap append cap %d", cap(got), cap(want))
		}
		if !slices.Equal(full, []int{1, 2}) {
			t.Errorf("Append past capacity modified the input: %v", full)
		}
	})

	t.Run("append-no-values", func(t *testing.T) {
		r := arena.Open()
		s := []int{1, 2, 3}
		got := arena.Append(r, s)
		if &got[0] != &s[0] {
			t.Errorf("Append(r, s) returned a different backing array")
		}
		if len(got) != len(s) || cap(got) != cap(s) {
			t.Errorf("Append(r, s) len %d cap %d, want %d %d", len(got), cap(got), len(s), cap(s))
		}
		none := arena.Append[int](r, nil)
		if none != nil {
			t.Errorf("Append(r, nil) = %v, want nil", none)
		}
	})
}
