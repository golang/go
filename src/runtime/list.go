// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

// listHead points to the head of an intrusive doubly-linked list.
//
// Prior to use, you must call init to store the offset of listNode fields.
//
// Every object in the list should be the same type.
type listHead struct {
	obj unsafe.Pointer

	initialized bool
	nodeOffset  uintptr
}

// init initializes the list head. off is the offset (via unsafe.Offsetof) of
// the listNode field in the objects in the list.
func (head *listHead) init(off uintptr) {
	head.initialized = true
	head.nodeOffset = off
}

// listNode is the linked list node for objects in a listHead list.
//
// listNode must be stored as a field in objects placed in the linked list. The
// offset of the field is registered via listHead.init.
//
// For example:
//
// type foo struct {
// 	val int
//
// 	node listNode
// }
//
// var fooHead listHead
// fooHead.init(unsafe.Offsetof(foo{}.node))
type listNode struct {
	prev unsafe.Pointer
	next unsafe.Pointer
}

func (head *listHead) getNode(p unsafe.Pointer) *listNode {
	if !head.initialized {
		throw("runtime: uninitialized listHead")
	}

	if p == nil {
		return nil
	}
	return (*listNode)(unsafe.Add(p, head.nodeOffset))
}

// Returns true if the list is empty.
func (head *listHead) empty() bool {
	return head.obj == nil
}

// Returns the head of the list without removing it.
func (head *listHead) head() unsafe.Pointer {
	return head.obj
}

// Push p onto the front of the list.
func (head *listHead) push(p unsafe.Pointer) {
	// p becomes the head of the list.

	// ... so p's next is the current head.
	pNode := head.getNode(p)
	pNode.next = head.obj

	// ... and the current head's prev is p.
	if head.obj != nil {
		headNode := head.getNode(head.obj)
		headNode.prev = p
	}

	head.obj = p
}

// Pop removes the head of the list.
func (head *listHead) pop() unsafe.Pointer {
	if head.obj == nil {
		return nil
	}

	// Return the head of the list.
	p := head.obj

	// ... so the new head is p's next.
	pNode := head.getNode(p)
	head.obj = pNode.next
	// p is no longer on the list. Clear next to remove unused references.
	// N.B. as the head, prev must already be nil.
	pNode.next = nil

	// ... and the new head no longer has a prev.
	if head.obj != nil {
		headNode := head.getNode(head.obj)
		headNode.prev = nil
	}

	return p
}

// Remove p from the middle of the list.
func (head *listHead) remove(p unsafe.Pointer) {
	if head.obj == p {
		// Use pop to ensure head is updated when removing the head.
		head.pop()
		return
	}

	pNode := head.getNode(p)
	prevNode := head.getNode(pNode.prev)
	nextNode := head.getNode(pNode.next)

	// Link prev to next.
	if prevNode != nil {
		prevNode.next = pNode.next
	}
	// Link next to prev.
	if nextNode != nil {
		nextNode.prev = pNode.prev
	}

	pNode.prev = nil
	pNode.next = nil
}
