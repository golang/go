// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/loader"
	"testing"
)

func TestHeap(t *testing.T) {
	tests := [][]loader.Sym{
		{10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
		{100, 90, 80, 70, 60, 50, 40, 30, 20, 10},
		{30, 50, 80, 20, 60, 70, 10, 100, 90, 40},
	}
	for _, s := range tests {
		h := heap{}
		for _, i := range s {
			h.push(i)
			if !verify(&h, 0) {
				t.Errorf("heap invariant violated: %v", h)
			}
		}
		for j := 0; j < len(s); j++ {
			x := h.pop()
			if !verify(&h, 0) {
				t.Errorf("heap invariant violated: %v", h)
			}
			// pop should return elements in ascending order.
			if want := loader.Sym((j + 1) * 10); x != want {
				t.Errorf("pop returns wrong element: want %d, got %d", want, x)
			}
		}
		if !h.empty() {
			t.Errorf("heap is not empty after all pops")
		}
	}

	// Also check that mixed pushes and pops work correctly.
	for _, s := range tests {
		h := heap{}
		for i := 0; i < len(s)/2; i++ {
			// two pushes, one pop
			h.push(s[2*i])
			if !verify(&h, 0) {
				t.Errorf("heap invariant violated: %v", h)
			}
			h.push(s[2*i+1])
			if !verify(&h, 0) {
				t.Errorf("heap invariant violated: %v", h)
			}
			h.pop()
			if !verify(&h, 0) {
				t.Errorf("heap invariant violated: %v", h)
			}
		}
		for !h.empty() { // pop remaining elements
			h.pop()
			if !verify(&h, 0) {
				t.Errorf("heap invariant violated: %v", h)
			}
		}
	}
}

// recursively verify heap-ness, starting at element i.
func verify(h *heap, i int) bool {
	n := len(*h)
	c1 := 2*i + 1 // left child
	c2 := 2*i + 2 // right child
	if c1 < n {
		if (*h)[c1] < (*h)[i] {
			return false
		}
		if !verify(h, c1) {
			return false
		}
	}
	if c2 < n {
		if (*h)[c2] < (*h)[i] {
			return false
		}
		if !verify(h, c2) {
			return false
		}
	}
	return true
}
