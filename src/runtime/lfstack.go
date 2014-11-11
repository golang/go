// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Lock-free stack.
// The following code runs only on g0 stack.

package runtime

import "unsafe"

const (
	// lfPtrBits and lfCountMask are defined in lfstack_*.go.
	lfPtrMask = 1<<lfPtrBits - 1
)

func lfstackpush(head *uint64, node *lfnode) {
	unode := uintptr(unsafe.Pointer(node))
	if unode&^lfPtrMask != 0 {
		print("p=", node, "\n")
		gothrow("lfstackpush: invalid pointer")
	}

	node.pushcnt++
	new := uint64(unode) | (uint64(node.pushcnt)&lfCountMask)<<lfPtrBits
	for {
		old := atomicload64(head)
		node.next = (*lfnode)(unsafe.Pointer(uintptr(old & lfPtrMask)))
		if cas64(head, old, new) {
			break
		}
	}
}

func lfstackpop(head *uint64) unsafe.Pointer {
	for {
		old := atomicload64(head)
		if old == 0 {
			return nil
		}
		node := (*lfnode)(unsafe.Pointer(uintptr(old & lfPtrMask)))
		node2 := (*lfnode)(atomicloadp(unsafe.Pointer(&node.next)))
		new := uint64(0)
		if node2 != nil {
			new = uint64(uintptr(unsafe.Pointer(node2))) | uint64(node2.pushcnt&lfCountMask)<<lfPtrBits
		}
		if cas64(head, old, new) {
			return unsafe.Pointer(node)
		}
	}
}
