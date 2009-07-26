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

	// The contents of this list element.
	Value interface {};
}

// List represents a doubly linked list.
type List struct {
	front, back *Element;
	len int;
}

// Init initializes or clears a List.
func (l *List) Init() *List {
	l.front = nil;
	l.back = nil;
	l.len = 0;
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

func (l *List) insertFront(e *Element) {
	e.prev = nil;
	e.next = l.front;
	l.front = e;
	if e.next != nil {
		e.next.prev = e;
	} else {
		l.back = e;
	}
	l.len++;
}

func (l *List) insertBack(e *Element) {
	e.next = nil;
	e.prev = l.back;
	l.back = e;
	if e.prev != nil {
		e.prev.next = e;
	} else {
		l.front = e;
	}
	l.len++;
}

// PushFront inserts the value at the front of the list, and returns a new Element containing it.
func (l *List) PushFront(value interface {}) *Element {
	e := &Element{ nil, nil, value };
	l.insertFront(e);
	return e
}

// PushBack inserts the value at the back of the list, and returns a new Element containing it.
func (l *List) PushBack(value interface {}) *Element {
	e := &Element{ nil, nil, value };
	l.insertBack(e);
	return e
}

// MoveToFront moves the element to the front of the list.
func (l *List) MoveToFront(e *Element) {
	if l.front == e {
		return
	}
	l.Remove(e);
	l.insertFront(e);
}

// MoveToBack moves the element to the back of the list.
func (l *List) MoveToBack(e *Element) {
	if l.back == e {
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
