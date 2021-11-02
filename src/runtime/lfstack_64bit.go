// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x || wasm

package runtime

import "unsafe"

const (
	// addrBits is the number of bits needed to represent a virtual address.
	//
	// See heapAddrBits for a table of address space sizes on
	// various architectures. 48 bits is enough for all
	// architectures except s390x.
	//
	// On AMD64, virtual addresses are 48-bit (or 57-bit) numbers sign extended to 64.
	// We shift the address left 16 to eliminate the sign extended part and make
	// room in the bottom for the count.
	//
	// On s390x, virtual addresses are 64-bit. There's not much we
	// can do about this, so we just hope that the kernel doesn't
	// get to really high addresses and panic if it does.
	addrBits = 48

	// In addition to the 16 bits taken from the top, we can take 3 from the
	// bottom, because node must be pointer-aligned, giving a total of 19 bits
	// of count.
	cntBits = 64 - addrBits + 3

	// On AIX, 64-bit addresses are split into 36-bit segment number and 28-bit
	// offset in segment.  Segment numbers in the range 0x0A0000000-0x0AFFFFFFF(LSA)
	// are available for mmap.
	// We assume all lfnode addresses are from memory allocated with mmap.
	// We use one bit to distinguish between the two ranges.
	aixAddrBits = 57
	aixCntBits  = 64 - aixAddrBits + 3
)

func lfstackPack(node *lfnode, cnt uintptr) uint64 {
	if GOARCH == "ppc64" && GOOS == "aix" {
		return uint64(uintptr(unsafe.Pointer(node)))<<(64-aixAddrBits) | uint64(cnt&(1<<aixCntBits-1))
	}
	return uint64(uintptr(unsafe.Pointer(node)))<<(64-addrBits) | uint64(cnt&(1<<cntBits-1))
}

func lfstackUnpack(val uint64) *lfnode {
	if GOARCH == "amd64" {
		// amd64 systems can place the stack above the VA hole, so we need to sign extend
		// val before unpacking.
		return (*lfnode)(unsafe.Pointer(uintptr(int64(val) >> cntBits << 3)))
	}
	if GOARCH == "ppc64" && GOOS == "aix" {
		return (*lfnode)(unsafe.Pointer(uintptr((val >> aixCntBits << 3) | 0xa<<56)))
	}
	return (*lfnode)(unsafe.Pointer(uintptr(val >> cntBits << 3)))
}
