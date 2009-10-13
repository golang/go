// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package provides heap operations for any type that implements
// heap.Interface.
//
package heap

import "sort"

// Any type that implements heap.Interface may be used as a
// heap with the following invariants (established after Init
// has been called):
//
//	h.Less(i, j) for 0 <= i < h.Len() and j = 2*i+1 or 2*i+2 and j < h.Len()
//
type Interface interface {
	sort.Interface;
	Push(x interface{});
	Pop() interface{};
}


// A heaper must be initialized before any of the heap operations
// can be used. Init is idempotent with respect to the heap invariants
// and may be called whenever the heap invariants may have been invalidated.
// Its complexity is O(n*log(n)) where n = h.Len().
//
func Init(h Interface) {
	sort.Sort(h);
}


// Push pushes the element x onto the heap. The complexity is
// O(log(n)) where n = h.Len().
//
func Push(h Interface, x interface{}) {
	h.Push(x);
	up(h, h.Len() - 1);
}


// Pop removes the minimum element (according to Less) from the heap
// and returns it. The complexity is O(log(n)) where n = h.Len().
//
func Pop(h Interface) interface{} {
	n := h.Len() - 1;
	h.Swap(0, n);
	down(h, 0, n);
	return h.Pop();
}


// Remove removes the element at index i from the heap.
// The complexity is O(log(n)) where n = h.Len().
//
func Remove(h Interface, i int) interface{} {
	n := h.Len() - 1;
	if n != i {
		h.Swap(n, i);
		down(h, i, n);
		up(h, i);
	}
	return h.Pop();
}


func up(h Interface, j int) {
	for {
		i := (j-1)/2;
		if i == j || h.Less(i, j) {
			break;
		}
		h.Swap(i, j);
		j = i;
	}
}


func down(h Interface, i, n int) {
	for {
		j := 2*i + 1;
		if j >= n {
			break;
		}
		if j1 := j+1; j1 < n && !h.Less(j, j1) {
			j = j1;	// = 2*i + 2
		}
		if h.Less(i, j) {
			break;
		}
		h.Swap(i, j);
		i = j;
	}
}
