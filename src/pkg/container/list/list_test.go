// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package list

import (
	"testing"
)

func checkListPointers(t *testing.T, l *List, es []*Element) {
	if len(es) == 0 {
		if l.front != nil || l.back != nil {
			t.Errorf("l.front/l.back = %v/%v should be nil/nil", l.front, l.back)
		}
		return
	}

	if l.front != es[0] {
		t.Errorf("l.front = %v, want %v", l.front, es[0])
	}
	if last := es[len(es)-1]; l.back != last {
		t.Errorf("l.back = %v, want %v", l.back, last)
	}

	for i, e := range es {
		var e_prev, e_next *Element = nil, nil
		if i > 0 {
			e_prev = es[i-1]
		}
		if i < len(es)-1 {
			e_next = es[i+1]
		}
		if e.prev != e_prev {
			t.Errorf("elt #%d (%v) has prev=%v, want %v", i, e, e.prev, e_prev)
		}
		if e.next != e_next {
			t.Errorf("elt #%d (%v) has next=%v, want %v", i, e, e.next, e_next)
		}
	}
}

func checkListLen(t *testing.T, l *List, n int) {
	if an := l.Len(); an != n {
		t.Errorf("l.Len() = %d, want %d", an, n)
	}
}

func TestList(t *testing.T) {
	l := New()
	checkListPointers(t, l, []*Element{})
	checkListLen(t, l, 0)

	// Single element list
	e := l.PushFront("a")
	checkListLen(t, l, 1)
	checkListPointers(t, l, []*Element{e})
	l.MoveToFront(e)
	checkListPointers(t, l, []*Element{e})
	l.MoveToBack(e)
	checkListPointers(t, l, []*Element{e})
	checkListLen(t, l, 1)
	l.Remove(e)
	checkListPointers(t, l, []*Element{})
	checkListLen(t, l, 0)

	// Bigger list
	e2 := l.PushFront(2)
	e1 := l.PushFront(1)
	e3 := l.PushBack(3)
	e4 := l.PushBack("banana")
	checkListPointers(t, l, []*Element{e1, e2, e3, e4})
	checkListLen(t, l, 4)

	l.Remove(e2)
	checkListPointers(t, l, []*Element{e1, e3, e4})
	checkListLen(t, l, 3)

	l.MoveToFront(e3) // move from middle
	checkListPointers(t, l, []*Element{e3, e1, e4})

	l.MoveToFront(e1)
	l.MoveToBack(e3) // move from middle
	checkListPointers(t, l, []*Element{e1, e4, e3})

	l.MoveToFront(e3) // move from back
	checkListPointers(t, l, []*Element{e3, e1, e4})
	l.MoveToFront(e3) // should be no-op
	checkListPointers(t, l, []*Element{e3, e1, e4})

	l.MoveToBack(e3) // move from front
	checkListPointers(t, l, []*Element{e1, e4, e3})
	l.MoveToBack(e3) // should be no-op
	checkListPointers(t, l, []*Element{e1, e4, e3})

	e2 = l.InsertBefore(2, e1) // insert before front
	checkListPointers(t, l, []*Element{e2, e1, e4, e3})
	l.Remove(e2)
	e2 = l.InsertBefore(2, e4) // insert before middle
	checkListPointers(t, l, []*Element{e1, e2, e4, e3})
	l.Remove(e2)
	e2 = l.InsertBefore(2, e3) // insert before back
	checkListPointers(t, l, []*Element{e1, e4, e2, e3})
	l.Remove(e2)

	e2 = l.InsertAfter(2, e1) // insert after front
	checkListPointers(t, l, []*Element{e1, e2, e4, e3})
	l.Remove(e2)
	e2 = l.InsertAfter(2, e4) // insert after middle
	checkListPointers(t, l, []*Element{e1, e4, e2, e3})
	l.Remove(e2)
	e2 = l.InsertAfter(2, e3) // insert after back
	checkListPointers(t, l, []*Element{e1, e4, e3, e2})
	l.Remove(e2)

	// Check standard iteration.
	sum := 0
	for e := l.Front(); e != nil; e = e.Next() {
		if i, ok := e.Value.(int); ok {
			sum += i
		}
	}
	if sum != 4 {
		t.Errorf("sum over l.Iter() = %d, want 4", sum)
	}

	// Clear all elements by iterating
	var next *Element
	for e := l.Front(); e != nil; e = next {
		next = e.Next()
		l.Remove(e)
	}
	checkListPointers(t, l, []*Element{})
	checkListLen(t, l, 0)
}

func checkList(t *testing.T, l *List, es []interface{}) {
	if l.Len() != len(es) {
		t.Errorf("list has len=%v, want %v", l.Len(), len(es))
		return
	}
	i := 0
	for e := l.Front(); e != nil; e = e.Next() {
		le := e.Value.(int)
		if le != es[i] {
			t.Errorf("elt #%d has value=%v, want %v", i, le, es[i])
		}
		i++
	}
}

func TestExtending(t *testing.T) {
	l1 := New()
	l2 := New()

	l1.PushBack(1)
	l1.PushBack(2)
	l1.PushBack(3)

	l2.PushBack(4)
	l2.PushBack(5)

	l3 := New()
	l3.PushBackList(l1)
	checkList(t, l3, []interface{}{1, 2, 3})
	l3.PushBackList(l2)
	checkList(t, l3, []interface{}{1, 2, 3, 4, 5})

	l3 = New()
	l3.PushFrontList(l2)
	checkList(t, l3, []interface{}{4, 5})
	l3.PushFrontList(l1)
	checkList(t, l3, []interface{}{1, 2, 3, 4, 5})

	checkList(t, l1, []interface{}{1, 2, 3})
	checkList(t, l2, []interface{}{4, 5})

	l3 = New()
	l3.PushBackList(l1)
	checkList(t, l3, []interface{}{1, 2, 3})
	l3.PushBackList(l3)
	checkList(t, l3, []interface{}{1, 2, 3, 1, 2, 3})

	l3 = New()
	l3.PushFrontList(l1)
	checkList(t, l3, []interface{}{1, 2, 3})
	l3.PushFrontList(l3)
	checkList(t, l3, []interface{}{1, 2, 3, 1, 2, 3})

	l3 = New()
	l1.PushBackList(l3)
	checkList(t, l1, []interface{}{1, 2, 3})
	l1.PushFrontList(l3)
	checkList(t, l1, []interface{}{1, 2, 3})
}

func TestRemove(t *testing.T) {
	l := New()
	e1 := l.PushBack(1)
	e2 := l.PushBack(2)
	checkListPointers(t, l, []*Element{e1, e2})
	e := l.Front()
	l.Remove(e)
	checkListPointers(t, l, []*Element{e2})
	checkListLen(t, l, 1)
	l.Remove(e)
	checkListPointers(t, l, []*Element{e2})
	checkListLen(t, l, 1)
}
