// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import "iter"

// intrusiveList is an intrusive doubly-linked list.
//
// The list is circular: head.prev is the tail of the list.
//
// The list's link function must be initialized to a function which returns the
// intrusiveLink structure for a list element. Elements may be a part of any number
// of lists at the same time, so long as they use separate links for each list.
type intrusiveList[T any] struct {
	head *T
	size int
	link func(*T) *intrusiveLink[T]
}

type intrusiveLink[T any] struct {
	prev, next *T
}

func (list *intrusiveList[T]) pushBack(e *T) {
	el := list.link(e)
	if el.next != nil {
		panic("element already in list")
	}
	list.size++
	if list.head == nil {
		list.head = e
		el.next = e
		el.prev = e
		return
	}
	headl := list.link(list.head)
	tail := headl.prev
	el.next = list.head
	el.prev = tail
	list.link(tail).next = e
	headl.prev = e
}

func (list *intrusiveList[T]) remove(e *T) {
	el := list.link(e)
	if el.next == nil {
		return // not in the list
	}
	if el.next == e {
		list.head = nil // only element in the list
	} else {
		if list.head == e {
			list.head = el.next
		}
		list.link(el.prev).next = el.next
		list.link(el.next).prev = el.prev
	}
	el.next = nil
	el.prev = nil
	list.size--
}

func (list *intrusiveList[T]) all() iter.Seq[*T] {
	return func(yield func(*T) bool) {
		e := list.head
		if e == nil {
			return
		}
		for {
			next := list.link(e).next
			last := next == list.head
			if !yield(e) || last {
				return
			}
			e = next
		}
	}
}

func (list *intrusiveList[T]) contains(e *T) bool {
	return list.link(e).next != nil
}
