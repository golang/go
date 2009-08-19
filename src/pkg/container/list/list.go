// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The list package implements a doubly linked list.
package list

// Element is an element in the linked list.
type Element struct {
	// Next and previous pointers in the doubly-linked list of elements.
	// The front of the list has prev = nil, and the back has next = nil.
	next, prev *Element;

	// A unique ID for the list to which this element belongs.
	id *byte;

	// The contents of this list element.
	Value interface {};
}

// List represents a doubly linked list.
type List struct {
	front, back *Element;
	len int;
	id *byte;
}

// Init initializes or clears a List.
func (l *List) Init() *List {
	l.front = nil;
	l.back = nil;
	l.len = 0;
	l.id = new(byte);
	return l
}

// New returns an initialized list.
func New() *List {
	return new(List).Init()
}

// Front returns the first element in the list.
func (l *List) Front() *Element {
	return l.front
}

// Back returns the last element in the list.
func (l *List) Back() *Element {
	return l.back
}

// Remove removes the element from the list.
func (l *List) Remove(e *Element) {
	if e.id != l.id {
		return
	}
	if e.prev == nil {
		l.front = e.next;
	} else {
		e.prev.next = e.next;
	}
	if e.next == nil {
		l.back = e.prev;
	} else {
		e.next.prev = e.prev;
	}

	e.prev = nil;
	e.next = nil;
	l.len--;
}

func (l *List) insertBefore(e *Element, mark *Element) {
	if mark.prev == nil {
		// new front of the list
		l.front = e;
	} else {
		mark.prev.next = e;
	}
	e.prev = mark.prev;
	mark.prev = e;
	e.next = mark;
	l.len++;
}

func (l *List) insertAfter(e *Element, mark *Element) {
	if mark.next == nil {
		// new back of the list
		l.back = e;
	} else {
		mark.next.prev = e;
	}
	e.next = mark.next;
	mark.next = e;
	e.prev = mark;
	l.len++;
}

func (l *List) insertFront(e *Element) {
	if l.front == nil {
		// empty list
		l.front, l.back = e, e;
		e.prev, e.next = nil, nil;
		l.len = 1;
		return
	}
	l.insertBefore(e, l.front);
}

func (l *List) insertBack(e *Element) {
	if l.back == nil {
		// empty list
		l.front, l.back = e, e;
		e.prev, e.next = nil, nil;
		l.len = 1;
		return
	}
	l.insertAfter(e, l.back);
}

// PushFront inserts the value at the front of the list and returns a new Element containing the value.
func (l *List) PushFront(value interface {}) *Element {
	if l.id == nil {
		l.Init();
	}
	e := &Element{ nil, nil, l.id, value };
	l.insertFront(e);
	return e
}

// PushBack inserts the value at the back of the list and returns a new Element containing the value.
func (l *List) PushBack(value interface {}) *Element {
	if l.id == nil {
		l.Init();
	}
	e := &Element{ nil, nil, l.id, value };
	l.insertBack(e);
	return e
}

// InsertBefore inserts the value immediately before mark and returns a new Element containing the value.
func (l *List) InsertBefore(value interface {}, mark *Element) *Element {
	if mark.id != l.id {
		return nil
	}
	e := &Element{ nil, nil, l.id, value };
	l.insertBefore(e, mark);
	return e
}

// InsertAfter inserts the value immediately after mark and returns a new Element containing the value.
func (l *List) InsertAfter(value interface {}, mark *Element) *Element {
	if mark.id != l.id {
		return nil
	}
	e := &Element{ nil, nil, l.id, value };
	l.insertAfter(e, mark);
	return e
}

// MoveToFront moves the element to the front of the list.
func (l *List) MoveToFront(e *Element) {
	if e.id != l.id || l.front == e {
		return
	}
	l.Remove(e);
	l.insertFront(e);
}

// MoveToBack moves the element to the back of the list.
func (l *List) MoveToBack(e *Element) {
	if e.id != l.id || l.back == e {
		return
	}
	l.Remove(e);
	l.insertBack(e);
}

// Len returns the number of elements in the list.
func (l *List) Len() int {
	return l.len
}

func (l *List) iterate(c chan <- *Element) {
	var next *Element;
	for e := l.front; e != nil; e = next {
		// Save next in case reader of c changes e.
		next = e.next;
		c <- e;
	}
	close(c);
}

func (l *List) Iter() <-chan *Element {
	c := make(chan *Element);
	go l.iterate(c);
	return c
}
