// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "internal/goarch"

const (
	// PageWords is the number of pointer-words per page.
	PageWords = PageSize / goarch.PtrSize

	// A malloc header is functionally a single type pointer, but
	// we need to use 8 here to ensure 8-byte alignment of allocations
	// on 32-bit platforms. It's wasteful, but a lot of code relies on
	// 8-byte alignment for 8-byte atomics.
	MallocHeaderSize = 8

	// The minimum object size that has a malloc header, exclusive.
	//
	// The size of this value controls overheads from the malloc header.
	// The minimum size is bound by writeHeapBitsSmall, which assumes that the
	// pointer bitmap for objects of a size smaller than this doesn't cross
	// more than one pointer-word boundary. This sets an upper-bound on this
	// value at the number of bits in a uintptr, multiplied by the pointer
	// size in bytes.
	//
	// We choose a value here that has a natural cutover point in terms of memory
	// overheads. This value just happens to be the maximum possible value this
	// can be.
	//
	// A span with heap bits in it will have 128 bytes of heap bits on 64-bit
	// platforms, and 256 bytes of heap bits on 32-bit platforms. The first size
	// class where malloc headers match this overhead for 64-bit platforms is
	// 512 bytes (8 KiB / 512 bytes * 8 bytes-per-header = 128 bytes of overhead).
	// On 32-bit platforms, this same point is the 256 byte size class
	// (8 KiB / 256 bytes * 8 bytes-per-header = 256 bytes of overhead).
	//
	// Guaranteed to be exactly at a size class boundary. The reason this value is
	// an exclusive minimum is subtle. Suppose we're allocating a 504-byte object
	// and its rounded up to 512 bytes for the size class. If minSizeForMallocHeader
	// is 512 and an inclusive minimum, then a comparison against minSizeForMallocHeader
	// by the two values would produce different results. In other words, the comparison
	// would not be invariant to size-class rounding. Eschewing this property means a
	// more complex check or possibly storing additional state to determine whether a
	// span has malloc headers.
	MinSizeForMallocHeader = goarch.PtrSize * goarch.PtrBits

	// PageSize is the increment in which spans are managed.
	PageSize = 1 << PageShift
)
