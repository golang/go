// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"runtime"
	"testing"
)

var spanDesc = map[uintptr]struct {
	pages uintptr
	scav  bool
}{
	0xc0000000: {2, false},
	0xc0006000: {1, false},
	0xc0010000: {8, false},
	0xc0022000: {7, false},
	0xc0034000: {4, true},
	0xc0040000: {5, false},
	0xc0050000: {5, true},
	0xc0060000: {5000, false},
}

// Wrap the Treap one more time because go:notinheap doesn't
// actually follow a structure across package boundaries.
//
//go:notinheap
type treap struct {
	runtime.Treap
}

func maskMatchName(mask, match runtime.TreapIterType) string {
	return fmt.Sprintf("%0*b-%0*b", runtime.TreapIterBits, uint8(mask), runtime.TreapIterBits, uint8(match))
}

func TestTreapFilter(t *testing.T) {
	var iterTypes = [...]struct {
		mask, match runtime.TreapIterType
		filter      runtime.TreapIterFilter // expected filter
	}{
		{0, 0, 0xf},
		{runtime.TreapIterScav, 0, 0x5},
		{runtime.TreapIterScav, runtime.TreapIterScav, 0xa},
		{runtime.TreapIterScav | runtime.TreapIterHuge, runtime.TreapIterHuge, 0x4},
		{runtime.TreapIterScav | runtime.TreapIterHuge, 0, 0x1},
		{0, runtime.TreapIterScav, 0x0},
	}
	for _, it := range iterTypes {
		t.Run(maskMatchName(it.mask, it.match), func(t *testing.T) {
			if f := runtime.TreapFilter(it.mask, it.match); f != it.filter {
				t.Fatalf("got %#x, want %#x", f, it.filter)
			}
		})
	}
}

// This test ensures that the treap implementation in the runtime
// maintains all stated invariants after different sequences of
// insert, removeSpan, find, and erase. Invariants specific to the
// treap data structure are checked implicitly: after each mutating
// operation, treap-related invariants are checked for the entire
// treap.
func TestTreap(t *testing.T) {
	// Set up a bunch of spans allocated into mheap_.
	// Also, derive a set of typeCounts of each type of span
	// according to runtime.TreapIterType so we can verify against
	// them later.
	spans := make([]runtime.Span, 0, len(spanDesc))
	typeCounts := [1 << runtime.TreapIterBits][1 << runtime.TreapIterBits]int{}
	for base, de := range spanDesc {
		s := runtime.AllocSpan(base, de.pages, de.scav)
		defer s.Free()
		spans = append(spans, s)

		for i := runtime.TreapIterType(0); i < 1<<runtime.TreapIterBits; i++ {
			for j := runtime.TreapIterType(0); j < 1<<runtime.TreapIterBits; j++ {
				if s.MatchesIter(i, j) {
					typeCounts[i][j]++
				}
			}
		}
	}
	t.Run("TypeCountsSanity", func(t *testing.T) {
		// Just sanity check type counts for a few values.
		check := func(mask, match runtime.TreapIterType, count int) {
			tc := typeCounts[mask][match]
			if tc != count {
				name := maskMatchName(mask, match)
				t.Fatalf("failed a sanity check for mask/match %s counts: got %d, wanted %d", name, tc, count)
			}
		}
		check(0, 0, len(spanDesc))
		check(runtime.TreapIterScav, 0, 6)
		check(runtime.TreapIterScav, runtime.TreapIterScav, 2)
	})
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
	t.Run("FindFirstFit", func(t *testing.T) {
		// Run this 10 times, recreating the treap each time.
		// Because of the non-deterministic structure of a treap,
		// we'll be able to test different structures this way.
		for i := 0; i < 10; i++ {
			tr := runtime.Treap{}
			for _, s := range spans {
				tr.Insert(s)
			}
			i := tr.Find(5)
			if i.Span().Base() != 0xc0010000 {
				t.Fatalf("expected span at lowest address which could fit 5 pages, instead found span at %x", i.Span().Base())
			}
			for _, s := range spans {
				tr.RemoveSpan(s)
			}
		}
	})
	t.Run("Iterate", func(t *testing.T) {
		for mask := runtime.TreapIterType(0); mask < 1<<runtime.TreapIterBits; mask++ {
			for match := runtime.TreapIterType(0); match < 1<<runtime.TreapIterBits; match++ {
				iterName := maskMatchName(mask, match)
				t.Run(iterName, func(t *testing.T) {
					t.Run("StartToEnd", func(t *testing.T) {
						// Ensure progressing an iterator actually goes over the whole treap
						// from the start and that it iterates over the elements in order.
						// Furthermore, ensure that it only iterates over the relevant parts
						// of the treap.
						// Finally, ensures that Start returns a valid iterator.
						tr := treap{}
						for _, s := range spans {
							tr.Insert(s)
						}
						nspans := 0
						lastBase := uintptr(0)
						for i := tr.Start(mask, match); i.Valid(); i = i.Next() {
							nspans++
							if lastBase > i.Span().Base() {
								t.Fatalf("not iterating in correct order: encountered base %x before %x", lastBase, i.Span().Base())
							}
							lastBase = i.Span().Base()
							if !i.Span().MatchesIter(mask, match) {
								t.Fatalf("found non-matching span while iteration over mask/match %s: base %x", iterName, i.Span().Base())
							}
						}
						if nspans != typeCounts[mask][match] {
							t.Fatal("failed to iterate forwards over full treap")
						}
						for _, s := range spans {
							tr.RemoveSpan(s)
						}
					})
					t.Run("EndToStart", func(t *testing.T) {
						// See StartToEnd tests.
						tr := treap{}
						for _, s := range spans {
							tr.Insert(s)
						}
						nspans := 0
						lastBase := ^uintptr(0)
						for i := tr.End(mask, match); i.Valid(); i = i.Prev() {
							nspans++
							if lastBase < i.Span().Base() {
								t.Fatalf("not iterating in correct order: encountered base %x before %x", lastBase, i.Span().Base())
							}
							lastBase = i.Span().Base()
							if !i.Span().MatchesIter(mask, match) {
								t.Fatalf("found non-matching span while iteration over mask/match %s: base %x", iterName, i.Span().Base())
							}
						}
						if nspans != typeCounts[mask][match] {
							t.Fatal("failed to iterate backwards over full treap")
						}
						for _, s := range spans {
							tr.RemoveSpan(s)
						}
					})
				})
			}
		}
		t.Run("Prev", func(t *testing.T) {
			// Test the iterator invariant that i.prev().next() == i.
			tr := treap{}
			for _, s := range spans {
				tr.Insert(s)
			}
			i := tr.Start(0, 0).Next().Next()
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
			i := tr.Start(0, 0).Next().Next()
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
		i := tr.Start(0, 0).Next().Next().Next()
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
		for i := tr.Start(0, 0); i.Valid(); {
			n := i.Next()
			tr.Erase(i)
			i = n
		}
		if size := tr.Size(); size != 0 {
			t.Fatalf("should have emptied out treap, %d spans left", size)
		}
	})
}
