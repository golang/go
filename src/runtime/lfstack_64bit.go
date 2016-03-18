// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm64 mips64 mips64le ppc64 ppc64le s390x

package runtime

import "unsafe"

const (
	// addrBits is the number of bits needed to represent a virtual address.
	//
	// In Linux the user address space for each architecture is limited as
	// follows (taken from the processor.h file for the architecture):
	//
	// Architecture  Name              Maximum Value (exclusive)
	// ---------------------------------------------------------------------
	// arm64         TASK_SIZE_64      Depends on configuration.
	// ppc64{,le}    TASK_SIZE_USER64  0x400000000000UL (46 bit addresses)
	// mips64{,le}   TASK_SIZE64       0x010000000000UL (40 bit addresses)
	// s390x         TASK_SIZE         0x020000000000UL (41 bit addresses)
	//
	// These values may increase over time.
	addrBits = 48

	// In addition to the 16 bits taken from the top, we can take 3 from the
	// bottom, because node must be pointer-aligned, giving a total of 19 bits
	// of count.
	cntBits = 64 - addrBits + 3
)

func lfstackPack(node *lfnode, cnt uintptr) uint64 {
	return uint64(uintptr(unsafe.Pointer(node)))<<(64-addrBits) | uint64(cnt&(1<<cntBits-1))
}

func lfstackUnpack(val uint64) *lfnode {
	return (*lfnode)(unsafe.Pointer(uintptr(val >> cntBits << 3)))
}
