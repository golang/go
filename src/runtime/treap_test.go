// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
)

var spanDesc = map[uintptr]uintptr{
	0xc0000000: 2,
	0xc0006000: 1,
	0xc0010000: 8,
	0xc0022000: 7,
	0xc0034000: 4,
	0xc0040000: 5,
	0xc0050000: 5,
	0xc0060000: 5000,
}

// Wrap the Treap one more time because go:notinheap doesn't
// actually follow a structure across package boundaries.
//
//go:notinheap
type treap struct {
	runtime.Treap
}

// This test ensures that the treap implementation in the runtime
// maintains all stated invariants after different sequences of
// insert, removeSpan, find, and erase. Invariants specific to the
// treap data structure are checked implicitly: after each mutating
// operation, treap-related invariants are checked for the entire
// treap.
func TestTreap(t *testing.T) {
	// Set up a bunch of spans allocated into mheap_.
	spans := make([]runtime.Span, 0, len(spanDesc))
	for base, pages := range spanDesc {
		s := runtime.AllocSpan(base, pages)
		defer s.Free()
		spans = append(spans, s)
	}
	t.Run("Insert", func(t *testing.T) {
		tr := treap{}
		// Test just a very basic insert/remove for sanity.
		tr.Insert(spans[0])
		tr.RemoveSpan(spans[0])
	})
	t.Run("FindTrivial", func(t *testing.T) {
		tr := treap{}
		// Test just a very basic find operation for sanity.
		tr.Insert(spans[0])
		i := tr.Find(1)
		if i.Span() != spans[0] {
			t.Fatal("found unknown span in treap")
		}
		tr.RemoveSpan(spans[0])
	})
	t.Run("FindBestFit", func(t *testing.T) {
		// Run this 10 times, recreating the treap each time.
		// Because of the non-deterministic structure of a treap,
		// we'll be able to test different structures this way.
		for i := 0; i < 10; i++ {
			tr := treap{}
			for _, s := range spans {
				tr.Insert(s)
			}
			i := tr.Find(5)
			if i.Span().Pages() != 5 {
				t.Fatalf("expected span of size 5, got span of size %d", i.Span().Pages())
			} else if i.Span().Base() != 0xc0040000 {
				t.Fatalf("expected span to have the lowest base address, instead got base %x", i.Span().Base())
			}
			for _, s := range spans {
				tr.RemoveSpan(s)
			}
		}
	})
	t.Run("Iterate", func(t *testing.T) {
		t.Run("StartToEnd", func(t *testing.T) {
			// Ensure progressing an iterator actually goes over the whole treap
			// from the start and that it iterates over the elements in order.
			// Also ensures that Start returns a valid iterator.
			tr := treap{}
			for _, s := range spans {
				tr.Insert(s)
			}
			nspans := 0
			lastSize := uintptr(0)
			for i := tr.Start(); i.Valid(); i = i.Next() {
				nspans++
				if lastSize > i.Span().Pages() {
					t.Fatalf("not iterating in correct order: encountered size %d before %d", lastSize, i.Span().Pages())
				}
				lastSize = i.Span().Pages()
			}
			if nspans != len(spans) {
				t.Fatal("failed to iterate forwards over full treap")
			}
			for _, s := range spans {
				tr.RemoveSpan(s)
			}
		})
		t.Run("EndToStart", func(t *testing.T) {
			// Ensure progressing an iterator actually goes over the whole treap
			// from the end and that it iterates over the elements in reverse
			// order. Also ensures that End returns a valid iterator.
			tr := treap{}
			for _, s := range spans {
				tr.Insert(s)
			}
			nspans := 0
			lastSize := ^uintptr(0)
			for i := tr.End(); i.Valid(); i = i.Prev() {
				nspans++
				if lastSize < i.Span().Pages() {
					t.Fatalf("not iterating in correct order: encountered size %d before %d", lastSize, i.Span().Pages())
				}
				lastSize = i.Span().Pages()
			}
			if nspans != len(spans) {
				t.Fatal("failed to iterate backwards over full treap")
			}
			for _, s := range spans {
				tr.RemoveSpan(s)
			}
		})
		t.Run("Prev", func(t *testing.T) {
			// Test the iterator invariant that i.prev().next() == i.
			tr := treap{}
			for _, s := range spans {
				tr.Insert(s)
			}
			i := tr.Start().Next().Next()
			p := i.Prev()
			if !p.Valid() {
				t.Fatal("i.prev() is invalid")
			}
			if p.Next().Span() != i.Span() {
				t.Fatal("i.prev().next() != i")
			}
			for _, s := range spans {
				tr.RemoveSpan(s)
			}
		})
		t.Run("Next", func(t *testing.T) {
			// Test the iterator invariant that i.next().prev() == i.
			tr := treap{}
			for _, s := range spans {
				tr.Insert(s)
			}
			i := tr.Start().Next().Next()
			n := i.Next()
			if !n.Valid() {
				t.Fatal("i.next() is invalid")
			}
			if n.Prev().Span() != i.Span() {
				t.Fatal("i.next().prev() != i")
			}
			for _, s := range spans {
				tr.RemoveSpan(s)
			}
		})
	})
	t.Run("EraseOne", func(t *testing.T) {
		// Test that erasing one iterator correctly retains
		// all relationships between elements.
		tr := treap{}
		for _, s := range spans {
			tr.Insert(s)
		}
		i := tr.Start().Next().Next().Next()
		s := i.Span()
		n := i.Next()
		p := i.Prev()
		tr.Erase(i)
		if n.Prev().Span() != p.Span() {
			t.Fatal("p, n := i.Prev(), i.Next(); n.prev() != p after i was erased")
		}
		if p.Next().Span() != n.Span() {
			t.Fatal("p, n := i.Prev(), i.Next(); p.next() != n after i was erased")
		}
		tr.Insert(s)
		for _, s := range spans {
			tr.RemoveSpan(s)
		}
	})
	t.Run("EraseAll", func(t *testing.T) {
		// Test that erasing iterators actually removes nodes from the treap.
		tr := treap{}
		for _, s := range spans {
			tr.Insert(s)
		}
		for i := tr.Start(); i.Valid(); {
			n := i.Next()
			tr.Erase(i)
			i = n
		}
		if size := tr.Size(); size != 0 {
			t.Fatalf("should have emptied out treap, %d spans left", size)
		}
	})
}
