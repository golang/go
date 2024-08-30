// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"fmt"
	"strconv"
)

func TestList() {
	l := a.New[string]()
	a.CheckListPointers(l, []*(a.Element[string]){})

	// Single element list
	e := l.PushFront("a")
	a.CheckListPointers(l, []*(a.Element[string]){e})
	l.MoveToFront(e)
	a.CheckListPointers(l, []*(a.Element[string]){e})
	l.MoveToBack(e)
	a.CheckListPointers(l, []*(a.Element[string]){e})
	l.Remove(e)
	a.CheckListPointers(l, []*(a.Element[string]){})

	// Bigger list
	l2 := a.New[int]()
	e2 := l2.PushFront(2)
	e1 := l2.PushFront(1)
	e3 := l2.PushBack(3)
	e4 := l2.PushBack(600)
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e2, e3, e4})

	l2.Remove(e2)
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e3, e4})

	l2.MoveToFront(e3) // move from middle
	a.CheckListPointers(l2, []*(a.Element[int]){e3, e1, e4})

	l2.MoveToFront(e1)
	l2.MoveToBack(e3) // move from middle
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e4, e3})

	l2.MoveToFront(e3) // move from back
	a.CheckListPointers(l2, []*(a.Element[int]){e3, e1, e4})
	l2.MoveToFront(e3) // should be no-op
	a.CheckListPointers(l2, []*(a.Element[int]){e3, e1, e4})

	l2.MoveToBack(e3) // move from front
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e4, e3})
	l2.MoveToBack(e3) // should be no-op
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e4, e3})

	e2 = l2.InsertBefore(2, e1) // insert before front
	a.CheckListPointers(l2, []*(a.Element[int]){e2, e1, e4, e3})
	l2.Remove(e2)
	e2 = l2.InsertBefore(2, e4) // insert before middle
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e2, e4, e3})
	l2.Remove(e2)
	e2 = l2.InsertBefore(2, e3) // insert before back
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e4, e2, e3})
	l2.Remove(e2)

	e2 = l2.InsertAfter(2, e1) // insert after front
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e2, e4, e3})
	l2.Remove(e2)
	e2 = l2.InsertAfter(2, e4) // insert after middle
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e4, e2, e3})
	l2.Remove(e2)
	e2 = l2.InsertAfter(2, e3) // insert after back
	a.CheckListPointers(l2, []*(a.Element[int]){e1, e4, e3, e2})
	l2.Remove(e2)

	// Check standard iteration.
	sum := 0
	for e := l2.Front(); e != nil; e = e.Next() {
		sum += e.Value
	}
	if sum != 604 {
		panic(fmt.Sprintf("sum over l = %d, want 604", sum))
	}

	// Clear all elements by iterating
	var next *a.Element[int]
	for e := l2.Front(); e != nil; e = next {
		next = e.Next()
		l2.Remove(e)
	}
	a.CheckListPointers(l2, []*(a.Element[int]){})
}

func checkList[T comparable](l *a.List[T], es []interface{}) {
	if !a.CheckListLen(l, len(es)) {
		return
	}

	i := 0
	for e := l.Front(); e != nil; e = e.Next() {
		le := e.Value
		// Comparison between a generically-typed variable le and an interface.
		if le != es[i] {
			panic(fmt.Sprintf("elt[%d].Value = %v, want %v", i, le, es[i]))
		}
		i++
	}
}

func TestExtending() {
	l1 := a.New[int]()
	l2 := a.New[int]()

	l1.PushBack(1)
	l1.PushBack(2)
	l1.PushBack(3)

	l2.PushBack(4)
	l2.PushBack(5)

	l3 := a.New[int]()
	l3.PushBackList(l1)
	checkList(l3, []interface{}{1, 2, 3})
	l3.PushBackList(l2)
	checkList(l3, []interface{}{1, 2, 3, 4, 5})

	l3 = a.New[int]()
	l3.PushFrontList(l2)
	checkList(l3, []interface{}{4, 5})
	l3.PushFrontList(l1)
	checkList(l3, []interface{}{1, 2, 3, 4, 5})

	checkList(l1, []interface{}{1, 2, 3})
	checkList(l2, []interface{}{4, 5})

	l3 = a.New[int]()
	l3.PushBackList(l1)
	checkList(l3, []interface{}{1, 2, 3})
	l3.PushBackList(l3)
	checkList(l3, []interface{}{1, 2, 3, 1, 2, 3})

	l3 = a.New[int]()
	l3.PushFrontList(l1)
	checkList(l3, []interface{}{1, 2, 3})
	l3.PushFrontList(l3)
	checkList(l3, []interface{}{1, 2, 3, 1, 2, 3})

	l3 = a.New[int]()
	l1.PushBackList(l3)
	checkList(l1, []interface{}{1, 2, 3})
	l1.PushFrontList(l3)
	checkList(l1, []interface{}{1, 2, 3})
}

func TestRemove() {
	l := a.New[int]()
	e1 := l.PushBack(1)
	e2 := l.PushBack(2)
	a.CheckListPointers(l, []*(a.Element[int]){e1, e2})
	e := l.Front()
	l.Remove(e)
	a.CheckListPointers(l, []*(a.Element[int]){e2})
	l.Remove(e)
	a.CheckListPointers(l, []*(a.Element[int]){e2})
}

func TestIssue4103() {
	l1 := a.New[int]()
	l1.PushBack(1)
	l1.PushBack(2)

	l2 := a.New[int]()
	l2.PushBack(3)
	l2.PushBack(4)

	e := l1.Front()
	l2.Remove(e) // l2 should not change because e is not an element of l2
	if n := l2.Len(); n != 2 {
		panic(fmt.Sprintf("l2.Len() = %d, want 2", n))
	}

	l1.InsertBefore(8, e)
	if n := l1.Len(); n != 3 {
		panic(fmt.Sprintf("l1.Len() = %d, want 3", n))
	}
}

func TestIssue6349() {
	l := a.New[int]()
	l.PushBack(1)
	l.PushBack(2)

	e := l.Front()
	l.Remove(e)
	if e.Value != 1 {
		panic(fmt.Sprintf("e.value = %d, want 1", e.Value))
	}
	if e.Next() != nil {
		panic(fmt.Sprintf("e.Next() != nil"))
	}
	if e.Prev() != nil {
		panic(fmt.Sprintf("e.Prev() != nil"))
	}
}

func TestMove() {
	l := a.New[int]()
	e1 := l.PushBack(1)
	e2 := l.PushBack(2)
	e3 := l.PushBack(3)
	e4 := l.PushBack(4)

	l.MoveAfter(e3, e3)
	a.CheckListPointers(l, []*(a.Element[int]){e1, e2, e3, e4})
	l.MoveBefore(e2, e2)
	a.CheckListPointers(l, []*(a.Element[int]){e1, e2, e3, e4})

	l.MoveAfter(e3, e2)
	a.CheckListPointers(l, []*(a.Element[int]){e1, e2, e3, e4})
	l.MoveBefore(e2, e3)
	a.CheckListPointers(l, []*(a.Element[int]){e1, e2, e3, e4})

	l.MoveBefore(e2, e4)
	a.CheckListPointers(l, []*(a.Element[int]){e1, e3, e2, e4})
	e2, e3 = e3, e2

	l.MoveBefore(e4, e1)
	a.CheckListPointers(l, []*(a.Element[int]){e4, e1, e2, e3})
	e1, e2, e3, e4 = e4, e1, e2, e3

	l.MoveAfter(e4, e1)
	a.CheckListPointers(l, []*(a.Element[int]){e1, e4, e2, e3})
	e2, e3, e4 = e4, e2, e3

	l.MoveAfter(e2, e3)
	a.CheckListPointers(l, []*(a.Element[int]){e1, e3, e2, e4})
	e2, e3 = e3, e2
}

// Test PushFront, PushBack, PushFrontList, PushBackList with uninitialized a.List
func TestZeroList() {
	var l1 = new(a.List[int])
	l1.PushFront(1)
	checkList(l1, []interface{}{1})

	var l2 = new(a.List[int])
	l2.PushBack(1)
	checkList(l2, []interface{}{1})

	var l3 = new(a.List[int])
	l3.PushFrontList(l1)
	checkList(l3, []interface{}{1})

	var l4 = new(a.List[int])
	l4.PushBackList(l2)
	checkList(l4, []interface{}{1})
}

// Test that a list l is not modified when calling InsertBefore with a mark that is not an element of l.
func TestInsertBeforeUnknownMark() {
	var l a.List[int]
	l.PushBack(1)
	l.PushBack(2)
	l.PushBack(3)
	l.InsertBefore(1, new(a.Element[int]))
	checkList(&l, []interface{}{1, 2, 3})
}

// Test that a list l is not modified when calling InsertAfter with a mark that is not an element of l.
func TestInsertAfterUnknownMark() {
	var l a.List[int]
	l.PushBack(1)
	l.PushBack(2)
	l.PushBack(3)
	l.InsertAfter(1, new(a.Element[int]))
	checkList(&l, []interface{}{1, 2, 3})
}

// Test that a list l is not modified when calling MoveAfter or MoveBefore with a mark that is not an element of l.
func TestMoveUnknownMark() {
	var l1 a.List[int]
	e1 := l1.PushBack(1)

	var l2 a.List[int]
	e2 := l2.PushBack(2)

	l1.MoveAfter(e1, e2)
	checkList(&l1, []interface{}{1})
	checkList(&l2, []interface{}{2})

	l1.MoveBefore(e1, e2)
	checkList(&l1, []interface{}{1})
	checkList(&l2, []interface{}{2})
}

// Test the Transform function.
func TestTransform() {
	l1 := a.New[int]()
	l1.PushBack(1)
	l1.PushBack(2)
	l2 := a.Transform(l1, strconv.Itoa)
	checkList(l2, []interface{}{"1", "2"})
}

func main() {
	TestList()
	TestExtending()
	TestRemove()
	TestIssue4103()
	TestIssue6349()
	TestMove()
	TestZeroList()
	TestInsertBeforeUnknownMark()
	TestInsertAfterUnknownMark()
	TestTransform()
}
