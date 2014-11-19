// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Lock-free stack.
// The following code runs only on g0 stack.

package runtime

import "unsafe"

func lfstackpush(head *uint64, node *lfnode) {
	node.pushcnt++
	new := lfstackPack(node, node.pushcnt)
	if node1, _ := lfstackUnpack(new); node1 != node {
		println("runtime: lfstackpush invalid packing: node=", node, " cnt=", hex(node.pushcnt), " packed=", hex(new), " -> node=", node1, "\n")
		gothrow("lfstackpush")
	}
	for {
		old := atomicload64(head)
		node.next, _ = lfstackUnpack(old)
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
		node, _ := lfstackUnpack(old)
		node2 := (*lfnode)(atomicloadp(unsafe.Pointer(&node.next)))
		new := uint64(0)
		if node2 != nil {
			new = lfstackPack(node2, node2.pushcnt)
		}
		if cas64(head, old, new) {
			return unsafe.Pointer(node)
		}
	}
}
