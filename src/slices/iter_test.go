// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slices_test

import (
	"iter"
	"math/rand/v2"
	. "slices"
	"testing"
)

func TestAll(t *testing.T) {
	for size := 0; size < 10; size++ {
		var s []int
		for i := range size {
			s = append(s, i)
		}
		ei, ev := 0, 0
		cnt := 0
		for i, v := range All(s) {
			if i != ei || v != ev {
				t.Errorf("at iteration %d got %d, %d want %d, %d", cnt, i, v, ei, ev)
			}
			ei++
			ev++
			cnt++
		}
		if cnt != size {
			t.Errorf("read %d values expected %d", cnt, size)
		}
	}
}

func TestBackward(t *testing.T) {
	for size := 0; size < 10; size++ {
		var s []int
		for i := range size {
			s = append(s, i)
		}
		ei, ev := size-1, size-1
		cnt := 0
		for i, v := range Backward(s) {
			if i != ei || v != ev {
				t.Errorf("at iteration %d got %d, %d want %d, %d", cnt, i, v, ei, ev)
			}
			ei--
			ev--
			cnt++
		}
		if cnt != size {
			t.Errorf("read %d values expected %d", cnt, size)
		}
	}
}

func TestValues(t *testing.T) {
	for size := 0; size < 10; size++ {
		var s []int
		for i := range size {
			s = append(s, i)
		}
		ev := 0
		cnt := 0
		for v := range Values(s) {
			if v != ev {
				t.Errorf("at iteration %d got %d want %d", cnt, v, ev)
			}
			ev++
			cnt++
		}
		if cnt != size {
			t.Errorf("read %d values expected %d", cnt, size)
		}
	}
}

func testSeq(yield func(int) bool) {
	for i := 0; i < 10; i += 2 {
		if !yield(i) {
			return
		}
	}
}

var testSeqResult = []int{0, 2, 4, 6, 8}

func TestAppendSeq(t *testing.T) {
	s := AppendSeq([]int{1, 2}, testSeq)
	want := append([]int{1, 2}, testSeqResult...)
	if !Equal(s, want) {
		t.Errorf("got %v, want %v", s, want)
	}
}

func TestAppendSeqFunc(t *testing.T) {
	testSeq := func(yield func(int) bool) {
		for i := 0; i < 5; i++ {
			if !yield(i) {
				return
			}
		}
	}

	s := AppendSeqFunc([]int{1, 2}, testSeq, func(val int) int { return val * 2 })
	want := append([]int{1, 2}, testSeqResult...)
	if !Equal(s, want) {
		t.Errorf("got %v, want %v", s, want)
	}
}

func TestCollect(t *testing.T) {
	s := Collect(testSeq)
	want := testSeqResult
	if !Equal(s, want) {
		t.Errorf("got %v, want %v", s, want)
	}
}

func TestCollectFunc(t *testing.T) {
	testSeq := func(yield func(int) bool) {
		for i := 0; i < 5; i++ {
			if !yield(i) {
				return
			}
		}
	}
	s := CollectFunc(testSeq, func(val int) int { return val * 2 })
	want := testSeqResult
	if !Equal(s, want) {
		t.Errorf("got %v, want %v", s, want)
	}
}

var iterTests = [][]string{
	nil,
	{"a"},
	{"a", "b"},
	{"b", "a"},
	strs[:],
}

func TestValuesAppendSeq(t *testing.T) {
	for _, prefix := range iterTests {
		for _, s := range iterTests {
			got := AppendSeq(prefix, Values(s))
			want := append(prefix, s...)
			if !Equal(got, want) {
				t.Errorf("AppendSeq(%v, Values(%v)) == %v, want %v", prefix, s, got, want)
			}
		}
	}
}

func TestValuesCollect(t *testing.T) {
	for _, s := range iterTests {
		got := Collect(Values(s))
		if !Equal(got, s) {
			t.Errorf("Collect(Values(%v)) == %v, want %v", s, got, s)
		}
	}
}

func TestSorted(t *testing.T) {
	s := Sorted(Values(ints[:]))
	if !IsSorted(s) {
		t.Errorf("sorted %v", ints)
		t.Errorf("   got %v", s)
	}
}

func TestSortedFunc(t *testing.T) {
	s := SortedFunc(Values(ints[:]), func(a, b int) int { return a - b })
	if !IsSorted(s) {
		t.Errorf("sorted %v", ints)
		t.Errorf("   got %v", s)
	}
}

func TestSortedStableFunc(t *testing.T) {
	n, m := 1000, 100
	data := make(intPairs, n)
	for i := range data {
		data[i].a = rand.IntN(m)
	}
	data.initB()

	s := intPairs(SortedStableFunc(Values(data), intPairCmp))
	if !IsSortedFunc(s, intPairCmp) {
		t.Errorf("SortedStableFunc didn't sort %d ints", n)
	}
	if !s.inOrder(false) {
		t.Errorf("SortedStableFunc wasn't stable on %d ints", n)
	}

	// iterVal converts a Seq2 to a Seq.
	iterVal := func(seq iter.Seq2[int, intPair]) iter.Seq[intPair] {
		return func(yield func(intPair) bool) {
			for _, v := range seq {
				if !yield(v) {
					return
				}
			}
		}
	}

	s = intPairs(SortedStableFunc(iterVal(Backward(data)), intPairCmp))
	if !IsSortedFunc(s, intPairCmp) {
		t.Errorf("SortedStableFunc didn't sort %d reverse ints", n)
	}
	if !s.inOrder(true) {
		t.Errorf("SortedStableFunc wasn't stable on %d reverse ints", n)
	}
}

func TestChunk(t *testing.T) {
	cases := []struct {
		name   string
		s      []int
		n      int
		chunks [][]int
	}{
		{
			name:   "nil",
			s:      nil,
			n:      1,
			chunks: nil,
		},
		{
			name:   "empty",
			s:      []int{},
			n:      1,
			chunks: nil,
		},
		{
			name:   "short",
			s:      []int{1, 2},
			n:      3,
			chunks: [][]int{{1, 2}},
		},
		{
			name:   "one",
			s:      []int{1, 2},
			n:      2,
			chunks: [][]int{{1, 2}},
		},
		{
			name:   "even",
			s:      []int{1, 2, 3, 4},
			n:      2,
			chunks: [][]int{{1, 2}, {3, 4}},
		},
		{
			name:   "odd",
			s:      []int{1, 2, 3, 4, 5},
			n:      2,
			chunks: [][]int{{1, 2}, {3, 4}, {5}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var chunks [][]int
			for c := range Chunk(tc.s, tc.n) {
				chunks = append(chunks, c)
			}

			if !chunkEqual(chunks, tc.chunks) {
				t.Errorf("Chunk(%v, %d) = %v, want %v", tc.s, tc.n, chunks, tc.chunks)
			}

			if len(chunks) == 0 {
				return
			}

			// Verify that appending to the end of the first chunk does not
			// clobber the beginning of the next chunk.
			s := Clone(tc.s)
			chunks[0] = append(chunks[0], -1)
			if !Equal(s, tc.s) {
				t.Errorf("slice was clobbered: %v, want %v", s, tc.s)
			}
		})
	}
}

func TestChunkPanics(t *testing.T) {
	for _, test := range []struct {
		name string
		x    []struct{}
		n    int
	}{
		{
			name: "cannot be less than 1",
			x:    make([]struct{}, 0),
			n:    0,
		},
	} {
		if !panics(func() { _ = Chunk(test.x, test.n) }) {
			t.Errorf("Chunk %s: got no panic, want panic", test.name)
		}
	}
}

func TestChunkRange(t *testing.T) {
	// Verify Chunk iteration can be stopped.
	var got [][]int
	for c := range Chunk([]int{1, 2, 3, 4, -100}, 2) {
		if len(got) == 2 {
			// Found enough values, break early.
			break
		}

		got = append(got, c)
	}

	if want := [][]int{{1, 2}, {3, 4}}; !chunkEqual(got, want) {
		t.Errorf("Chunk iteration did not stop, got %v, want %v", got, want)
	}
}

func chunkEqual[Slice ~[]E, E comparable](s1, s2 []Slice) bool {
	return EqualFunc(s1, s2, Equal[Slice])
}
