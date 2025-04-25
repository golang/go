// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || loong64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x || wasm

package runtime

import (
	"internal/goarch"
	"internal/goos"
	"unsafe"
)

const (
	// addrBits is the number of bits needed to represent a virtual address.
	//
	// See heapAddrBits for a table of address space sizes on
	// various architectures. 48 bits is enough for all
	// arch/os combos except s390x, aix, and riscv64.
	//
	// On AMD64, virtual addresses are 48-bit (or 57-bit) sign-extended.
	// Other archs are 48-bit zero-extended.
	//
	// On s390x, virtual addresses are 64-bit. There's not much we
	// can do about this, so we just hope that the kernel doesn't
	// get to really high addresses and panic if it does.
	defaultAddrBits = 48

	// On AIX, 64-bit addresses are split into 36-bit segment number and 28-bit
	// offset in segment.  Segment numbers in the range 0x0A0000000-0x0AFFFFFFF(LSA)
	// are available for mmap.
	// We assume all tagged addresses are from memory allocated with mmap.
	// We use one bit to distinguish between the two ranges.
	aixAddrBits = 57

	// riscv64 SV57 mode gives 56 bits of userspace VA.
	// tagged pointer code supports it,
	// but broader support for SV57 mode is incomplete,
	// and there may be other issues (see #54104).
	riscv64AddrBits = 56

	addrBits = goos.IsAix*aixAddrBits + goarch.IsRiscv64*riscv64AddrBits + (1-goos.IsAix)*(1-goarch.IsRiscv64)*defaultAddrBits

	// In addition to the 16 bits (or other, depending on arch/os) taken from the top,
	// we can take 9 from the bottom, because we require pointers to be well-aligned
	// (see tagptr.go:tagAlignBits). That gives us a total of 25 bits for the tag.
	tagBits = 64 - addrBits + tagAlignBits
)

// taggedPointerPack created a taggedPointer from a pointer and a tag.
// Tag bits that don't fit in the result are discarded.
func taggedPointerPack(ptr unsafe.Pointer, tag uintptr) taggedPointer {
	t := taggedPointer(uint64(uintptr(ptr))<<(tagBits-tagAlignBits) | uint64(tag&(1<<tagBits-1)))
	if t.pointer() != ptr || t.tag() != tag {
		print("runtime: taggedPointerPack invalid packing: ptr=", ptr, " tag=", hex(tag), " packed=", hex(t), " -> ptr=", t.pointer(), " tag=", hex(t.tag()), "\n")
		throw("taggedPointerPack")
	}
	return t
}

// Pointer returns the pointer from a taggedPointer.
func (tp taggedPointer) pointer() unsafe.Pointer {
	if GOARCH == "amd64" {
		// amd64 systems can place the stack above the VA hole, so we need to sign extend
		// val before unpacking.
		return unsafe.Pointer(uintptr(int64(tp) >> tagBits << tagAlignBits))
	}
	if GOOS == "aix" {
		return unsafe.Pointer(uintptr((tp >> tagBits << tagAlignBits) | 0xa<<56))
	}
	return unsafe.Pointer(uintptr(tp >> tagBits << tagAlignBits))
}

// Tag returns the tag from a taggedPointer.
func (tp taggedPointer) tag() uintptr {
	return uintptr(tp & (1<<tagBits - 1))
}
