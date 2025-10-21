// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

// The types in this file are exact copies of the types in list.go, but with
// unsafe.Pointer replaced with uintptr for use where write barriers must be
// avoided, such as uses of muintptr, puintptr, guintptr.
//
// Objects in these lists must be kept alive via another real reference.

// listHeadManual points to the head of an intrusive doubly-linked list of
// objects.
//
// Prior to use, you must call init to store the offset of listNodeManual fields.
//
// Every object in the list should be the same type.
type listHeadManual struct {
	obj uintptr

	initialized bool
	nodeOffset  uintptr
}

// init initializes the list head. off is the offset (via unsafe.Offsetof) of
// the listNodeManual field in the objects in the list.
func (head *listHeadManual) init(off uintptr) {
	head.initialized = true
	head.nodeOffset = off
}

// listNodeManual is the linked list node for objects in a listHeadManual list.
//
// listNodeManual must be stored as a field in objects placed in the linked list.
// The offset of the field is registered via listHeadManual.init.
//
// For example:
//
// type foo struct {
// 	val int
//
// 	node listNodeManual
// }
//
// var fooHead listHeadManual
// fooHead.init(unsafe.Offsetof(foo{}.node))
type listNodeManual struct {
	prev uintptr
	next uintptr
}

func (head *listHeadManual) getNode(p unsafe.Pointer) *listNodeManual {
	if !head.initialized {
		throw("runtime: uninitialized listHead")
	}

	if p == nil {
		return nil
	}
	return (*listNodeManual)(unsafe.Add(p, head.nodeOffset))
}

// Returns true if the list is empty.
func (head *listHeadManual) empty() bool {
	return head.obj == 0
}

// Returns the head of the list without removing it.
func (head *listHeadManual) head() unsafe.Pointer {
	return unsafe.Pointer(head.obj)
}

// Push p onto the front of the list.
func (head *listHeadManual) push(p unsafe.Pointer) {
	// p becomes the head of the list.

	// ... so p's next is the current head.
	pNode := head.getNode(p)
	pNode.next = head.obj

	// ... and the current head's prev is p.
	if head.obj != 0 {
		headNode := head.getNode(unsafe.Pointer(head.obj))
		headNode.prev = uintptr(p)
	}

	head.obj = uintptr(p)
}

// Pop removes the head of the list.
func (head *listHeadManual) pop() unsafe.Pointer {
	if head.obj == 0 {
		return nil
	}

	// Return the head of the list.
	p := unsafe.Pointer(head.obj)

	// ... so the new head is p's next.
	pNode := head.getNode(p)
	head.obj = pNode.next
	// p is no longer on the list. Clear next to remove unused references.
	// N.B. as the head, prev must already be nil.
	pNode.next = 0

	// ... and the new head no longer has a prev.
	if head.obj != 0 {
		headNode := head.getNode(unsafe.Pointer(head.obj))
		headNode.prev = 0
	}

	return p
}

// Remove p from the middle of the list.
func (head *listHeadManual) remove(p unsafe.Pointer) {
	if unsafe.Pointer(head.obj) == p {
		// Use pop to ensure head is updated when removing the head.
		head.pop()
		return
	}

	pNode := head.getNode(p)
	prevNode := head.getNode(unsafe.Pointer(pNode.prev))
	nextNode := head.getNode(unsafe.Pointer(pNode.next))

	// Link prev to next.
	if prevNode != nil {
		prevNode.next = pNode.next
	}
	// Link next to prev.
	if nextNode != nil {
		nextNode.prev = pNode.prev
	}

	pNode.prev = 0
	pNode.next = 0
}
