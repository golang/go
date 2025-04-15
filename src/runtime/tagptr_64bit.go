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

	// In addition to the 16 bits taken from the top, we can take 9 from the
	// bottom, because we require pointers to be well-aligned (see tagptr.go:tagAlignBits).
	// That gives us a total of 25 bits for the tag.
	tagBits = 64 - addrBits + tagAlignBits

	// On AIX, 64-bit addresses are split into 36-bit segment number and 28-bit
	// offset in segment.  Segment numbers in the range 0x0A0000000-0x0AFFFFFFF(LSA)
	// are available for mmap.
	// We assume all tagged addresses are from memory allocated with mmap.
	// We use one bit to distinguish between the two ranges.
	aixAddrBits = 57
	aixTagBits  = 64 - aixAddrBits + tagAlignBits

	// riscv64 SV57 mode gives 56 bits of userspace VA.
	// tagged pointer code supports it,
	// but broader support for SV57 mode is incomplete,
	// and there may be other issues (see #54104).
	riscv64AddrBits = 56
	riscv64TagBits  = 64 - riscv64AddrBits + tagAlignBits
)

// The number of bits stored in the numeric tag of a taggedPointer
const taggedPointerBits = (goos.IsAix * aixTagBits) + (goarch.IsRiscv64 * riscv64TagBits) + ((1 - goos.IsAix) * (1 - goarch.IsRiscv64) * tagBits)

// taggedPointerPack created a taggedPointer from a pointer and a tag.
// Tag bits that don't fit in the result are discarded.
func taggedPointerPack(ptr unsafe.Pointer, tag uintptr) taggedPointer {
	var t taggedPointer
	if GOOS == "aix" {
		if GOARCH != "ppc64" {
			throw("check this code for aix on non-ppc64")
		}
		t = taggedPointer(uint64(uintptr(ptr))<<(64-aixAddrBits) | uint64(tag&(1<<aixTagBits-1)))
	} else if GOARCH == "riscv64" {
		t = taggedPointer(uint64(uintptr(ptr))<<(64-riscv64AddrBits) | uint64(tag&(1<<riscv64TagBits-1)))
	} else {
		t = taggedPointer(uint64(uintptr(ptr))<<(64-addrBits) | uint64(tag&(1<<tagBits-1)))
	}
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
		return unsafe.Pointer(uintptr((tp >> aixTagBits << tagAlignBits) | 0xa<<56))
	}
	if GOARCH == "riscv64" {
		return unsafe.Pointer(uintptr(tp >> riscv64TagBits << tagAlignBits))
	}
	return unsafe.Pointer(uintptr(tp >> tagBits << tagAlignBits))
}

// Tag returns the tag from a taggedPointer.
func (tp taggedPointer) tag() uintptr {
	return uintptr(tp & (1<<taggedPointerBits - 1))
}
