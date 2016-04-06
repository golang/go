// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Lock-free stack.
// The following code runs only on g0 stack.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

func lfstackpush(head *uint64, node *lfnode) {
	node.pushcnt++
	new := lfstackPack(node, node.pushcnt)
	if node1 := lfstackUnpack(new); node1 != node {
		print("runtime: lfstackpush invalid packing: node=", node, " cnt=", hex(node.pushcnt), " packed=", hex(new), " -> node=", node1, "\n")
		throw("lfstackpush")
	}
	for {
		old := atomic.Load64(head)
		node.next = old
		if atomic.Cas64(head, old, new) {
			break
		}
	}
}

func lfstackpop(head *uint64) unsafe.Pointer {
	for {
		old := atomic.Load64(head)
		if old == 0 {
			return nil
		}
		node := lfstackUnpack(old)
		next := atomic.Load64(&node.next)
		if atomic.Cas64(head, old, next) {
			return unsafe.Pointer(node)
		}
	}
}

const (
	addrBits = 48
	cntBits  = 64 - addrBits + 3
)

func lfstackPack(node *lfnode, cnt uintptr) uint64 {
	if unsafe.Sizeof(uintptr(0)) == 4 {
		// On 32-bit systems, the stored uint64 has a 32-bit pointer and 32-bit count.
		return uint64(uintptr(unsafe.Pointer(node)))<<32 | uint64(cnt)
	}
	// On ppc64, Linux limits the user address space to 46 bits (see
	// TASK_SIZE_USER64 in the Linux kernel).  This has grown over time,
	// so here we allow 48 bit addresses.
	//
	// On mips64, Linux limits the user address space to 40 bits (see
	// TASK_SIZE64 in the Linux kernel).  This has grown over time,
	// so here we allow 48 bit addresses.
	//
	// On AMD64, virtual addresses are 48-bit numbers sign extended to 64.
	// We shift the address left 16 to eliminate the sign extended part and make
	// room in the bottom for the count.
	//
	// In addition to the 16 bits taken from the top, we can take 3 from the
	// bottom, because node must be pointer-aligned, giving a total of 19 bits
	// of count.
	return uint64(uintptr(unsafe.Pointer(node)))<<(64-addrBits) | uint64(cnt&(1<<cntBits-1))
}

func lfstackUnpack(val uint64) *lfnode {
	if unsafe.Sizeof(uintptr(0)) == 4 {
		return (*lfnode)(unsafe.Pointer(uintptr(val >> 32)))
	}
	return (*lfnode)(unsafe.Pointer(uintptr(val >> cntBits << 3)))
}
