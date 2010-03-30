// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package heap

import (
	"testing"
	"container/vector"
)


type myHeap struct {
	// A vector.Vector implements sort.Interface except for Less,
	// and it implements Push and Pop as required for heap.Interface.
	vector.Vector
}


func (h *myHeap) Less(i, j int) bool { return h.At(i).(int) < h.At(j).(int) }


func (h *myHeap) verify(t *testing.T, i int) {
	n := h.Len()
	j1 := 2*i + 1
	j2 := 2*i + 2
	if j1 < n {
		if h.Less(j1, i) {
			t.Errorf("heap invariant invalidated [%d] = %d > [%d] = %d", i, h.At(i), j1, h.At(j1))
			return
		}
		h.verify(t, j1)
	}
	if j2 < n {
		if h.Less(j2, i) {
			t.Errorf("heap invariant invalidated [%d] = %d > [%d] = %d", i, h.At(i), j1, h.At(j2))
			return
		}
		h.verify(t, j2)
	}
}


func TestInit0(t *testing.T) {
	h := new(myHeap)
	for i := 20; i > 0; i-- {
		h.Push(0) // all elements are the same
	}
	Init(h)
	h.verify(t, 0)

	for i := 1; h.Len() > 0; i++ {
		x := Pop(h).(int)
		h.verify(t, 0)
		if x != 0 {
			t.Errorf("%d.th pop got %d; want %d", i, x, 0)
		}
	}
}


func TestInit1(t *testing.T) {
	h := new(myHeap)
	for i := 20; i > 0; i-- {
		h.Push(i) // all elements are different
	}
	Init(h)
	h.verify(t, 0)

	for i := 1; h.Len() > 0; i++ {
		x := Pop(h).(int)
		h.verify(t, 0)
		if x != i {
			t.Errorf("%d.th pop got %d; want %d", i, x, i)
		}
	}
}


func Test(t *testing.T) {
	h := new(myHeap)
	h.verify(t, 0)

	for i := 20; i > 10; i-- {
		h.Push(i)
	}
	Init(h)
	h.verify(t, 0)

	for i := 10; i > 0; i-- {
		Push(h, i)
		h.verify(t, 0)
	}

	for i := 1; h.Len() > 0; i++ {
		x := Pop(h).(int)
		if i < 20 {
			Push(h, 20+i)
		}
		h.verify(t, 0)
		if x != i {
			t.Errorf("%d.th pop got %d; want %d", i, x, i)
		}
	}
}


func TestRemove0(t *testing.T) {
	h := new(myHeap)
	for i := 0; i < 10; i++ {
		h.Push(i)
	}
	h.verify(t, 0)

	for h.Len() > 0 {
		i := h.Len() - 1
		x := Remove(h, i).(int)
		if x != i {
			t.Errorf("Remove(%d) got %d; want %d", i, x, i)
		}
		h.verify(t, 0)
	}
}


func TestRemove1(t *testing.T) {
	h := new(myHeap)
	for i := 0; i < 10; i++ {
		h.Push(i)
	}
	h.verify(t, 0)

	for i := 0; h.Len() > 0; i++ {
		x := Remove(h, 0).(int)
		if x != i {
			t.Errorf("Remove(0) got %d; want %d", x, i)
		}
		h.verify(t, 0)
	}
}


func TestRemove2(t *testing.T) {
	N := 10

	h := new(myHeap)
	for i := 0; i < N; i++ {
		h.Push(i)
	}
	h.verify(t, 0)

	m := make(map[int]bool)
	for h.Len() > 0 {
		m[Remove(h, (h.Len()-1)/2).(int)] = true
		h.verify(t, 0)
	}

	if len(m) != N {
		t.Errorf("len(m) = %d; want %d", len(m), N)
	}
	for i := 0; i < len(m); i++ {
		if !m[i] {
			t.Errorf("m[%d] doesn't exist", i)
		}
	}
}
