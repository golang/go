// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import "testing"

func TestQueue(t *testing.T) {
	// Zero valued queue should have 0 length and capacity.
	var q queue
	if n := q.len; n != 0 {
		t.Fatalf("empty queue has len %d; want 0", n)
	}
	if n := q.cap(); n != 0 {
		t.Fatalf("empty queue has cap %d; want 0", n)
	}

	// As we add elements, len should grow.
	N := 32
	for i := 0; i < N; i++ {
		q.enqueue(i)
		if n := q.len; n != i+1 {
			t.Fatalf("after adding %d elements, queue has len %d", i, n)
		}
		if v, ok := q.peek(); !ok {
			t.Fatalf("couldn't peek after adding %d elements", i)
		} else if v.(int) != 0 {
			t.Fatalf("after adding %d elements, peek is %d; want 0", i, v)
		}
	}

	// As we remove and add elements, len should shrink and grow.
	// We should also remove elements in the same order they were added.
	want := 0
	for _, r := range []int{1, 2, 3, 5, 8, 13, 21} {
		s := make([]int, 0, r)
		for i := 0; i < r; i++ {
			if got, ok := q.dequeue(); !ok {
				t.Fatalf("after removing %d of %d elements, could not dequeue", i+1, r)
			} else if got != want {
				t.Fatalf("after removing %d of %d elements, got %d; want %d", i+1, r, got, want)
			} else {
				s = append(s, got.(int))
			}
			want = (want + 1) % N
			if n := q.len; n != N-i-1 {
				t.Fatalf("after removing %d of %d elements, len is %d; want %d", i+1, r, n, N-i-1)
			}
		}
		for i, v := range s {
			q.enqueue(v)
			if n := q.len; n != N-r+i+1 {
				t.Fatalf("after adding back %d of %d elements, len is %d; want %d", i+1, r, n, n-r+i+1)
			}
		}
	}
}
