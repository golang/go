// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le
// +build linux

package runtime

import "unsafe"

// On ppc64, Linux limits the user address space to 46 bits (see
// TASK_SIZE_USER64 in the Linux kernel).  This has grown over time,
// so here we allow 48 bit addresses.
//
// In addition to the 16 bits taken from the top, we can take 3 from the
// bottom, because node must be pointer-aligned, giving a total of 19 bits
// of count.
const (
	addrBits = 48
	cntBits  = 64 - addrBits + 3
)

func lfstackPack(node *lfnode, cnt uintptr) uint64 {
	return uint64(uintptr(unsafe.Pointer(node)))<<(64-addrBits) | uint64(cnt&(1<<cntBits-1))
}

func lfstackUnpack(val uint64) (node *lfnode, cnt uintptr) {
	node = (*lfnode)(unsafe.Pointer(uintptr(val >> cntBits << 3)))
	cnt = uintptr(val & (1<<cntBits - 1))
	return
}
