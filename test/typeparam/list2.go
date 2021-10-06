// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package list provides a doubly linked list of some element type
// (generic form of the "container/list" package).

package main

import (
	"fmt"
	"strconv"
)

// Element is an element of a linked list.
type _Element[T any] struct {
	// Next and previous pointers in the doubly-linked list of elements.
	// To simplify the implementation, internally a list l is implemented
	// as a ring, such that &l.root is both the next element of the last
	// list element (l.Back()) and the previous element of the first list
	// element (l.Front()).
	next, prev *_Element[T]

	// The list to which this element belongs.
	list *_List[T]

	// The value stored with this element.
	Value T
}

// Next returns the next list element or nil.
func (e *_Element[T]) Next() *_Element[T] {
	if p := e.next; e.list != nil && p != &e.list.root {
		return p
	}
	return nil
}

// Prev returns the previous list element or nil.
func (e *_Element[T]) Prev() *_Element[T] {
	if p := e.prev; e.list != nil && p != &e.list.root {
		return p
	}
	return nil
}

// _List represents a doubly linked list.
// The zero value for _List is an empty list ready to use.
type _List[T any] struct {
	root _Element[T] // sentinel list element, only &root, root.prev, and root.next are used
	len  int         // current list length excluding (this) sentinel element
}

// Init initializes or clears list l.
func (l *_List[T]) Init() *_List[T] {
	l.root.next = &l.root
	l.root.prev = &l.root
	l.len = 0
	return l
}

// New returns an initialized list.
func _New[T any]() *_List[T] { return new(_List[T]).Init() }

// Len returns the number of elements of list l.
// The complexity is O(1).
func (l *_List[_]) Len() int { return l.len }

// Front returns the first element of list l or nil if the list is empty.
func (l *_List[T]) Front() *_Element[T] {
	if l.len == 0 {
		return nil
	}
	return l.root.next
}

// Back returns the last element of list l or nil if the list is empty.
func (l *_List[T]) Back() *_Element[T] {
	if l.len == 0 {
		return nil
	}
	return l.root.prev
}

// lazyInit lazily initializes a zero _List value.
func (l *_List[_]) lazyInit() {
	if l.root.next == nil {
		l.Init()
	}
}

// insert inserts e after at, increments l.len, and returns e.
func (l *_List[T]) insert(e, at *_Element[T]) *_Element[T] {
	e.prev = at
	e.next = at.next
	e.prev.next = e
	e.next.prev = e
	e.list = l
	l.len++
	return e
}

// insertValue is a convenience wrapper for insert(&_Element[T]{Value: v}, at).
func (l *_List[T]) insertValue(v T, at *_Element[T]) *_Element[T] {
	return l.insert(&_Element[T]{Value: v}, at)
}

// remove removes e from its list, decrements l.len, and returns e.
func (l *_List[T]) remove(e *_Element[T]) *_Element[T] {
	e.prev.next = e.next
	e.next.prev = e.prev
	e.next = nil // avoid memory leaks
	e.prev = nil // avoid memory leaks
	e.list = nil
	l.len--
	return e
}

// move moves e to next to at and returns e.
func (l *_List[T]) move(e, at *_Element[T]) *_Element[T] {
	if e == at {
		return e
	}
	e.prev.next = e.next
	e.next.prev = e.prev

	e.prev = at
	e.next = at.next
	e.prev.next = e
	e.next.prev = e

	return e
}

// Remove removes e from l if e is an element of list l.
// It returns the element value e.Value.
// The element must not be nil.
func (l *_List[T]) Remove(e *_Element[T]) T {
	if e.list == l {
		// if e.list == l, l must have been initialized when e was inserted
		// in l or l == nil (e is a zero _Element) and l.remove will crash
		l.remove(e)
	}
	return e.Value
}

// PushFront inserts a new element e with value v at the front of list l and returns e.
func (l *_List[T]) PushFront(v T) *_Element[T] {
	l.lazyInit()
	return l.insertValue(v, &l.root)
}

// PushBack inserts a new element e with value v at the back of list l and returns e.
func (l *_List[T]) PushBack(v T) *_Element[T] {
	l.lazyInit()
	return l.insertValue(v, l.root.prev)
}

// InsertBefore inserts a new element e with value v immediately before mark and returns e.
// If mark is not an element of l, the list is not modified.
// The mark must not be nil.
func (l *_List[T]) InsertBefore(v T, mark *_Element[T]) *_Element[T] {
	if mark.list != l {
		return nil
	}
	// see comment in _List.Remove about initialization of l
	return l.insertValue(v, mark.prev)
}

// InsertAfter inserts a new element e with value v immediately after mark and returns e.
// If mark is not an element of l, the list is not modified.
// The mark must not be nil.
func (l *_List[T]) InsertAfter(v T, mark *_Element[T]) *_Element[T] {
	if mark.list != l {
		return nil
	}
	// see comment in _List.Remove about initialization of l
	return l.insertValue(v, mark)
}

// MoveToFront moves element e to the front of list l.
// If e is not an element of l, the list is not modified.
// The element must not be nil.
func (l *_List[T]) MoveToFront(e *_Element[T]) {
	if e.list != l || l.root.next == e {
		return
	}
	// see comment in _List.Remove about initialization of l
	l.move(e, &l.root)
}

// MoveToBack moves element e to the back of list l.
// If e is not an element of l, the list is not modified.
// The element must not be nil.
func (l *_List[T]) MoveToBack(e *_Element[T]) {
	if e.list != l || l.root.prev == e {
		return
	}
	// see comment in _List.Remove about initialization of l
	l.move(e, l.root.prev)
}

// MoveBefore moves element e to its new position before mark.
// If e or mark is not an element of l, or e == mark, the list is not modified.
// The element and mark must not be nil.
func (l *_List[T]) MoveBefore(e, mark *_Element[T]) {
	if e.list != l || e == mark || mark.list != l {
		return
	}
	l.move(e, mark.prev)
}

// MoveAfter moves element e to its new position after mark.
// If e or mark is not an element of l, or e == mark, the list is not modified.
// The element and mark must not be nil.
func (l *_List[T]) MoveAfter(e, mark *_Element[T]) {
	if e.list != l || e == mark || mark.list != l {
		return
	}
	l.move(e, mark)
}

// PushBackList inserts a copy of an other list at the back of list l.
// The lists l and other may be the same. They must not be nil.
func (l *_List[T]) PushBackList(other *_List[T]) {
	l.lazyInit()
	for i, e := other.Len(), other.Front(); i > 0; i, e = i-1, e.Next() {
		l.insertValue(e.Value, l.root.prev)
	}
}

// PushFrontList inserts a copy of an other list at the front of list l.
// The lists l and other may be the same. They must not be nil.
func (l *_List[T]) PushFrontList(other *_List[T]) {
	l.lazyInit()
	for i, e := other.Len(), other.Back(); i > 0; i, e = i-1, e.Prev() {
		l.insertValue(e.Value, &l.root)
	}
}

// Transform runs a transform function on a list returning a new list.
func _Transform[TElem1, TElem2 any](lst *_List[TElem1], f func(TElem1) TElem2) *_List[TElem2] {
	ret := _New[TElem2]()
	for p := lst.Front(); p != nil; p = p.Next() {
		ret.PushBack(f(p.Value))
	}
	return ret
}

func checkListLen[T any](l *_List[T], len int) bool {
	if n := l.Len(); n != len {
		panic(fmt.Sprintf("l.Len() = %d, want %d", n, len))
		return false
	}
	return true
}

func checkListPointers[T any](l *_List[T], es []*_Element[T]) {
	root := &l.root

	if !checkListLen(l, len(es)) {
		return
	}

	// zero length lists must be the zero value or properly initialized (sentinel circle)
	if len(es) == 0 {
		if l.root.next != nil && l.root.next != root || l.root.prev != nil && l.root.prev != root {
			panic(fmt.Sprintf("l.root.next = %p, l.root.prev = %p; both should both be nil or %p", l.root.next, l.root.prev, root))
		}
		return
	}
	// len(es) > 0

	// check internal and external prev/next connections
	for i, e := range es {
		prev := root
		Prev := (*_Element[T])(nil)
		if i > 0 {
			prev = es[i-1]
			Prev = prev
		}
		if p := e.prev; p != prev {
			panic(fmt.Sprintf("elt[%d](%p).prev = %p, want %p", i, e, p, prev))
		}
		if p := e.Prev(); p != Prev {
			panic(fmt.Sprintf("elt[%d](%p).Prev() = %p, want %p", i, e, p, Prev))
		}

		next := root
		Next := (*_Element[T])(nil)
		if i < len(es)-1 {
			next = es[i+1]
			Next = next
		}
		if n := e.next; n != next {
			panic(fmt.Sprintf("elt[%d](%p).next = %p, want %p", i, e, n, next))
		}
		if n := e.Next(); n != Next {
			panic(fmt.Sprintf("elt[%d](%p).Next() = %p, want %p", i, e, n, Next))
		}
	}
}

func TestList() {
	l := _New[string]()
	checkListPointers(l, []*(_Element[string]){})

	// Single element list
	e := l.PushFront("a")
	checkListPointers(l, []*(_Element[string]){e})
	l.MoveToFront(e)
	checkListPointers(l, []*(_Element[string]){e})
	l.MoveToBack(e)
	checkListPointers(l, []*(_Element[string]){e})
	l.Remove(e)
	checkListPointers(l, []*(_Element[string]){})

	// Bigger list
	l2 := _New[int]()
	e2 := l2.PushFront(2)
	e1 := l2.PushFront(1)
	e3 := l2.PushBack(3)
	e4 := l2.PushBack(600)
	checkListPointers(l2, []*(_Element[int]){e1, e2, e3, e4})

	l2.Remove(e2)
	checkListPointers(l2, []*(_Element[int]){e1, e3, e4})

	l2.MoveToFront(e3) // move from middle
	checkListPointers(l2, []*(_Element[int]){e3, e1, e4})

	l2.MoveToFront(e1)
	l2.MoveToBack(e3) // move from middle
	checkListPointers(l2, []*(_Element[int]){e1, e4, e3})

	l2.MoveToFront(e3) // move from back
	checkListPointers(l2, []*(_Element[int]){e3, e1, e4})
	l2.MoveToFront(e3) // should be no-op
	checkListPointers(l2, []*(_Element[int]){e3, e1, e4})

	l2.MoveToBack(e3) // move from front
	checkListPointers(l2, []*(_Element[int]){e1, e4, e3})
	l2.MoveToBack(e3) // should be no-op
	checkListPointers(l2, []*(_Element[int]){e1, e4, e3})

	e2 = l2.InsertBefore(2, e1) // insert before front
	checkListPointers(l2, []*(_Element[int]){e2, e1, e4, e3})
	l2.Remove(e2)
	e2 = l2.InsertBefore(2, e4) // insert before middle
	checkListPointers(l2, []*(_Element[int]){e1, e2, e4, e3})
	l2.Remove(e2)
	e2 = l2.InsertBefore(2, e3) // insert before back
	checkListPointers(l2, []*(_Element[int]){e1, e4, e2, e3})
	l2.Remove(e2)

	e2 = l2.InsertAfter(2, e1) // insert after front
	checkListPointers(l2, []*(_Element[int]){e1, e2, e4, e3})
	l2.Remove(e2)
	e2 = l2.InsertAfter(2, e4) // insert after middle
	checkListPointers(l2, []*(_Element[int]){e1, e4, e2, e3})
	l2.Remove(e2)
	e2 = l2.InsertAfter(2, e3) // insert after back
	checkListPointers(l2, []*(_Element[int]){e1, e4, e3, e2})
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
	var next *_Element[int]
	for e := l2.Front(); e != nil; e = next {
		next = e.Next()
		l2.Remove(e)
	}
	checkListPointers(l2, []*(_Element[int]){})
}

func checkList[T comparable](l *_List[T], es []interface{}) {
	if !checkListLen(l, len(es)) {
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
	l1 := _New[int]()
	l2 := _New[int]()

	l1.PushBack(1)
	l1.PushBack(2)
	l1.PushBack(3)

	l2.PushBack(4)
	l2.PushBack(5)

	l3 := _New[int]()
	l3.PushBackList(l1)
	checkList(l3, []interface{}{1, 2, 3})
	l3.PushBackList(l2)
	checkList(l3, []interface{}{1, 2, 3, 4, 5})

	l3 = _New[int]()
	l3.PushFrontList(l2)
	checkList(l3, []interface{}{4, 5})
	l3.PushFrontList(l1)
	checkList(l3, []interface{}{1, 2, 3, 4, 5})

	checkList(l1, []interface{}{1, 2, 3})
	checkList(l2, []interface{}{4, 5})

	l3 = _New[int]()
	l3.PushBackList(l1)
	checkList(l3, []interface{}{1, 2, 3})
	l3.PushBackList(l3)
	checkList(l3, []interface{}{1, 2, 3, 1, 2, 3})

	l3 = _New[int]()
	l3.PushFrontList(l1)
	checkList(l3, []interface{}{1, 2, 3})
	l3.PushFrontList(l3)
	checkList(l3, []interface{}{1, 2, 3, 1, 2, 3})

	l3 = _New[int]()
	l1.PushBackList(l3)
	checkList(l1, []interface{}{1, 2, 3})
	l1.PushFrontList(l3)
	checkList(l1, []interface{}{1, 2, 3})
}

func TestRemove() {
	l := _New[int]()
	e1 := l.PushBack(1)
	e2 := l.PushBack(2)
	checkListPointers(l, []*(_Element[int]){e1, e2})
	e := l.Front()
	l.Remove(e)
	checkListPointers(l, []*(_Element[int]){e2})
	l.Remove(e)
	checkListPointers(l, []*(_Element[int]){e2})
}

func TestIssue4103() {
	l1 := _New[int]()
	l1.PushBack(1)
	l1.PushBack(2)

	l2 := _New[int]()
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
	l := _New[int]()
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
	l := _New[int]()
	e1 := l.PushBack(1)
	e2 := l.PushBack(2)
	e3 := l.PushBack(3)
	e4 := l.PushBack(4)

	l.MoveAfter(e3, e3)
	checkListPointers(l, []*(_Element[int]){e1, e2, e3, e4})
	l.MoveBefore(e2, e2)
	checkListPointers(l, []*(_Element[int]){e1, e2, e3, e4})

	l.MoveAfter(e3, e2)
	checkListPointers(l, []*(_Element[int]){e1, e2, e3, e4})
	l.MoveBefore(e2, e3)
	checkListPointers(l, []*(_Element[int]){e1, e2, e3, e4})

	l.MoveBefore(e2, e4)
	checkListPointers(l, []*(_Element[int]){e1, e3, e2, e4})
	e2, e3 = e3, e2

	l.MoveBefore(e4, e1)
	checkListPointers(l, []*(_Element[int]){e4, e1, e2, e3})
	e1, e2, e3, e4 = e4, e1, e2, e3

	l.MoveAfter(e4, e1)
	checkListPointers(l, []*(_Element[int]){e1, e4, e2, e3})
	e2, e3, e4 = e4, e2, e3

	l.MoveAfter(e2, e3)
	checkListPointers(l, []*(_Element[int]){e1, e3, e2, e4})
	e2, e3 = e3, e2
}

// Test PushFront, PushBack, PushFrontList, PushBackList with uninitialized _List
func TestZeroList() {
	var l1 = new(_List[int])
	l1.PushFront(1)
	checkList(l1, []interface{}{1})

	var l2 = new(_List[int])
	l2.PushBack(1)
	checkList(l2, []interface{}{1})

	var l3 = new(_List[int])
	l3.PushFrontList(l1)
	checkList(l3, []interface{}{1})

	var l4 = new(_List[int])
	l4.PushBackList(l2)
	checkList(l4, []interface{}{1})
}

// Test that a list l is not modified when calling InsertBefore with a mark that is not an element of l.
func TestInsertBeforeUnknownMark() {
	var l _List[int]
	l.PushBack(1)
	l.PushBack(2)
	l.PushBack(3)
	l.InsertBefore(1, new(_Element[int]))
	checkList(&l, []interface{}{1, 2, 3})
}

// Test that a list l is not modified when calling InsertAfter with a mark that is not an element of l.
func TestInsertAfterUnknownMark() {
	var l _List[int]
	l.PushBack(1)
	l.PushBack(2)
	l.PushBack(3)
	l.InsertAfter(1, new(_Element[int]))
	checkList(&l, []interface{}{1, 2, 3})
}

// Test that a list l is not modified when calling MoveAfter or MoveBefore with a mark that is not an element of l.
func TestMoveUnknownMark() {
	var l1 _List[int]
	e1 := l1.PushBack(1)

	var l2 _List[int]
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
	l1 := _New[int]()
	l1.PushBack(1)
	l1.PushBack(2)
	l2 := _Transform(l1, strconv.Itoa)
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
