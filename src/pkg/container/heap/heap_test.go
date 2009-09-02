// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package heap

import (
	"testing";
	"container/vector";
)


type myHeap struct {
	vector.IntVector;
}


func newHeap() *myHeap {
	var h myHeap;
	h.IntVector.Init(0);
	return &h;
}


func (h *myHeap) verify(t *testing.T, i int) {
	n := h.Len();
	j1 := 2*i + 1;
	j2 := 2*i + 2;
	if j1 < n {
		if h.Less(j1, i) {
			t.Errorf("heap invariant invalidated [%d] = %d > [%d] = %d", i, h.At(i), j1, h.At(j1));
			return;
		}
		h.verify(t, j1);
	}
	if j2 < n {
		if h.Less(j2, i) {
			t.Errorf("heap invariant invalidated [%d] = %d > [%d] = %d", i, h.At(i), j1, h.At(j2));
			return;
		}
		h.verify(t, j2);
	}
}


func (h *myHeap) Push(x interface{}) {
	h.IntVector.Push(x.(int));
}


func (h *myHeap) Pop() interface{} {
	return h.IntVector.Pop();
}


func TestInit(t *testing.T) {
	h := newHeap();
	for i := 20; i > 0; i-- {
		h.Push(i);
	}
	Init(h);
	h.verify(t, 0);

	for i := 1; h.Len() > 0; i++ {
		x := Pop(h).(int);
		h.verify(t, 0);
		if x != i {
			t.Errorf("%d.th pop got %d; want %d", i, x, i);
		}
	}
}


func Test(t *testing.T) {
	h := newHeap();
	h.verify(t, 0);

	for i := 20; i > 10; i-- {
		h.Push(i);
	}
	Init(h);
	h.verify(t, 0);

	for i := 10; i > 0; i-- {
		Push(h, i);
		h.verify(t, 0);
	}

	for i := 1; h.Len() > 0; i++ {
		x := Pop(h).(int);
		if i < 20 {
			Push(h, 20+i);
		}
		h.verify(t, 0);
		if x != i {
			t.Errorf("%d.th pop got %d; want %d", i, x, i);
		}
	}
}
