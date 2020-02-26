// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 arm mips mipsle wasm darwin,arm64

// wasm is a treated as a 32-bit architecture for the purposes of the page
// allocator, even though it has 64-bit pointers. This is because any wasm
// pointer always has its top 32 bits as zero, so the effective heap address
// space is only 2^32 bytes in size (see heapAddrBits).

// darwin/arm64 is treated as a 32-bit architecture for the purposes of the
// page allocator, even though it has 64-bit pointers and a 33-bit address
// space (see heapAddrBits). The 33 bit address space cannot be rounded up
// to 64 bits because there are too many summary levels to fit in just 33
// bits.

package runtime

import "unsafe"

const (
	// The number of levels in the radix tree.
	summaryLevels = 4

	// Constants for testing.
	pageAlloc32Bit = 1
	pageAlloc64Bit = 0

	// Number of bits needed to represent all indices into the L1 of the
	// chunks map.
	//
	// See (*pageAlloc).chunks for more details. Update the documentation
	// there should this number change.
	pallocChunksL1Bits = 0
)

// See comment in mpagealloc_64bit.go.
var levelBits = [summaryLevels]uint{
	summaryL0Bits,
	summaryLevelBits,
	summaryLevelBits,
	summaryLevelBits,
}

// See comment in mpagealloc_64bit.go.
var levelShift = [summaryLevels]uint{
	heapAddrBits - summaryL0Bits,
	heapAddrBits - summaryL0Bits - 1*summaryLevelBits,
	heapAddrBits - summaryL0Bits - 2*summaryLevelBits,
	heapAddrBits - summaryL0Bits - 3*summaryLevelBits,
}

// See comment in mpagealloc_64bit.go.
var levelLogPages = [summaryLevels]uint{
	logPallocChunkPages + 3*summaryLevelBits,
	logPallocChunkPages + 2*summaryLevelBits,
	logPallocChunkPages + 1*summaryLevelBits,
	logPallocChunkPages,
}

// See mpagealloc_64bit.go for details.
func (s *pageAlloc) sysInit() {
	// Calculate how much memory all our entries will take up.
	//
	// This should be around 12 KiB or less.
	totalSize := uintptr(0)
	for l := 0; l < summaryLevels; l++ {
		totalSize += (uintptr(1) << (heapAddrBits - levelShift[l])) * pallocSumBytes
	}
	totalSize = alignUp(totalSize, physPageSize)

	// Reserve memory for all levels in one go. There shouldn't be much for 32-bit.
	reservation := sysReserve(nil, totalSize)
	if reservation == nil {
		throw("failed to reserve page summary memory")
	}
	// There isn't much. Just map it and mark it as used immediately.
	sysMap(reservation, totalSize, s.sysStat)
	sysUsed(reservation, totalSize)

	// Iterate over the reservation and cut it up into slices.
	//
	// Maintain i as the byte offset from reservation where
	// the new slice should start.
	for l, shift := range levelShift {
		entries := 1 << (heapAddrBits - shift)

		// Put this reservation into a slice.
		sl := notInHeapSlice{(*notInHeap)(reservation), 0, entries}
		s.summary[l] = *(*[]pallocSum)(unsafe.Pointer(&sl))

		reservation = add(reservation, uintptr(entries)*pallocSumBytes)
	}
}

// See mpagealloc_64bit.go for details.
func (s *pageAlloc) sysGrow(base, limit uintptr) {
	if base%pallocChunkBytes != 0 || limit%pallocChunkBytes != 0 {
		print("runtime: base = ", hex(base), ", limit = ", hex(limit), "\n")
		throw("sysGrow bounds not aligned to pallocChunkBytes")
	}

	// Walk up the tree and update the summary slices.
	for l := len(s.summary) - 1; l >= 0; l-- {
		// Figure out what part of the summary array this new address space needs.
		// Note that we need to align the ranges to the block width (1<<levelBits[l])
		// at this level because the full block is needed to compute the summary for
		// the next level.
		lo, hi := addrsToSummaryRange(l, base, limit)
		_, hi = blockAlignSummaryRange(l, lo, hi)
		if hi > len(s.summary[l]) {
			s.summary[l] = s.summary[l][:hi]
		}
	}
}
