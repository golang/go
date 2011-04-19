// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package list implements a doubly linked list.
//
// To iterate over a list (where l is a *List):
//	for e := l.Front(); e != nil; e = e.Next() {
//		// do something with e.Value
//	}
//
package list

// Element is an element in the linked list.
type Element struct {
	// Next and previous pointers in the doubly-linked list of elements.
	// The front of the list has prev = nil, and the back has next = nil.
	next, prev *Element

	// The list to which this element belongs.
	list *List

	// The contents of this list element.
	Value interface{}
}

// Next returns the next list element or nil.
func (e *Element) Next() *Element { return e.next }

// Prev returns the previous list element or nil.
func (e *Element) Prev() *Element { return e.prev }

// List represents a doubly linked list.
// The zero value for List is an empty list ready to use.
type List struct {
	front, back *Element
	len         int
}

// Init initializes or clears a List.
func (l *List) Init() *List {
	l.front = nil
	l.back = nil
	l.len = 0
	return l
}

// New returns an initialized list.
func New() *List { return new(List) }

// Front returns the first element in the list.
func (l *List) Front() *Element { return l.front }

// Back returns the last element in the list.
func (l *List) Back() *Element { return l.back }

// Remove removes the element from the list
// and returns its Value.
func (l *List) Remove(e *Element) interface{} {
	l.remove(e)
	e.list = nil // do what remove does not
	return e.Value
}

// remove the element from the list, but do not clear the Element's list field.
// This is so that other List methods may use remove when relocating Elements
// without needing to restore the list field.
func (l *List) remove(e *Element) {
	if e.list != l {
		return
	}
	if e.prev == nil {
		l.front = e.next
	} else {
		e.prev.next = e.next
	}
	if e.next == nil {
		l.back = e.prev
	} else {
		e.next.prev = e.prev
	}

	e.prev = nil
	e.next = nil
	l.len--
}

func (l *List) insertBefore(e *Element, mark *Element) {
	if mark.prev == nil {
		// new front of the list
		l.front = e
	} else {
		mark.prev.next = e
	}
	e.prev = mark.prev
	mark.prev = e
	e.next = mark
	l.len++
}

func (l *List) insertAfter(e *Element, mark *Element) {
	if mark.next == nil {
		// new back of the list
		l.back = e
	} else {
		mark.next.prev = e
	}
	e.next = mark.next
	mark.next = e
	e.prev = mark
	l.len++
}

func (l *List) insertFront(e *Element) {
	if l.front == nil {
		// empty list
		l.front, l.back = e, e
		e.prev, e.next = nil, nil
		l.len = 1
		return
	}
	l.insertBefore(e, l.front)
}

func (l *List) insertBack(e *Element) {
	if l.back == nil {
		// empty list
		l.front, l.back = e, e
		e.prev, e.next = nil, nil
		l.len = 1
		return
	}
	l.insertAfter(e, l.back)
}

// PushFront inserts the value at the front of the list and returns a new Element containing the value.
func (l *List) PushFront(value interface{}) *Element {
	e := &Element{nil, nil, l, value}
	l.insertFront(e)
	return e
}

// PushBack inserts the value at the back of the list and returns a new Element containing the value.
func (l *List) PushBack(value interface{}) *Element {
	e := &Element{nil, nil, l, value}
	l.insertBack(e)
	return e
}

// InsertBefore inserts the value immediately before mark and returns a new Element containing the value.
func (l *List) InsertBefore(value interface{}, mark *Element) *Element {
	if mark.list != l {
		return nil
	}
	e := &Element{nil, nil, l, value}
	l.insertBefore(e, mark)
	return e
}

// InsertAfter inserts the value immediately after mark and returns a new Element containing the value.
func (l *List) InsertAfter(value interface{}, mark *Element) *Element {
	if mark.list != l {
		return nil
	}
	e := &Element{nil, nil, l, value}
	l.insertAfter(e, mark)
	return e
}

// MoveToFront moves the element to the front of the list.
func (l *List) MoveToFront(e *Element) {
	if e.list != l || l.front == e {
		return
	}
	l.remove(e)
	l.insertFront(e)
}

// MoveToBack moves the element to the back of the list.
func (l *List) MoveToBack(e *Element) {
	if e.list != l || l.back == e {
		return
	}
	l.remove(e)
	l.insertBack(e)
}

// Len returns the number of elements in the list.
func (l *List) Len() int { return l.len }

// PushBackList inserts each element of ol at the back of the list.
func (l *List) PushBackList(ol *List) {
	last := ol.Back()
	for e := ol.Front(); e != nil; e = e.Next() {
		l.PushBack(e.Value)
		if e == last {
			break
		}
	}
}

// PushFrontList inserts each element of ol at the front of the list. The ordering of the passed list is preserved.
func (l *List) PushFrontList(ol *List) {
	first := ol.Front()
	for e := ol.Back(); e != nil; e = e.Prev() {
		l.PushFront(e.Value)
		if e == first {
			break
		}
	}
}
