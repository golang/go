// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Page allocator.
//
// The page allocator manages mapped pages (defined by pageSize, NOT
// physPageSize) for allocation and re-use. It is embedded into mheap.
//
// Pages are managed using a bitmap that is sharded into chunks.
// In the bitmap, 1 means in-use, and 0 means free. The bitmap spans the
// process's address space. Chunks are allocated using a SLAB allocator
// and pointers to chunks are managed in one large array, which is mapped
// in as needed.
//
// The bitmap is efficiently searched by using a radix tree in combination
// with fast bit-wise intrinsics. Allocation is performed using an address-ordered
// first-fit approach.
//
// Each entry in the radix tree is a summary that describes three properties of
// a particular region of the address space: the number of contiguous free pages
// at the start and end of the region it represents, and the maximum number of
// contiguous free pages found anywhere in that region.
//
// Each level of the radix tree is stored as one contiguous array, which represents
// a different granularity of subdivision of the processes' address space. Thus, this
// radix tree is actually implicit in these large arrays, as opposed to having explicit
// dynamically-allocated pointer-based node structures. Naturally, these arrays may be
// quite large for system with large address spaces, so in these cases they are mapped
// into memory as needed. The leaf summaries of the tree correspond to a bitmap chunk.
//
// The root level (referred to as L0 and index 0 in pageAlloc.summary) has each
// summary represent the largest section of address space (16 GiB on 64-bit systems),
// with each subsequent level representing successively smaller subsections until we
// reach the finest granularity at the leaves, a chunk.
//
// More specifically, each summary in each level (except for leaf summaries)
// represents some number of entries in the following level. For example, each
// summary in the root level may represent a 16 GiB region of address space,
// and in the next level there could be 8 corresponding entries which represent 2
// GiB subsections of that 16 GiB region, each of which could correspond to 8
// entries in the next level which each represent 256 MiB regions, and so on.
//
// Thus, this design only scales to heaps so large, but can always be extended to
// larger heaps by simply adding levels to the radix tree, which mostly costs
// additional virtual address space. The choice of managing large arrays also means
// that a large amount of virtual address space may be reserved by the runtime.

package runtime

const (
	// The size of a bitmap chunk, i.e. the amount of bits (that is, pages) to consider
	// in the bitmap at once.
	pallocChunkPages    = 1 << logPallocChunkPages
	pallocChunkBytes    = pallocChunkPages * pageSize
	logPallocChunkPages = 9
	logPallocChunkBytes = logPallocChunkPages + pageShift

	// The number of radix bits for each level.
	//
	// The value of 3 is chosen such that the block of summaries we need to scan at
	// each level fits in 64 bytes (2^3 summaries * 8 bytes per summary), which is
	// close to the L1 cache line width on many systems. Also, a value of 3 fits 4 tree
	// levels perfectly into the 21-bit mallocBits summary field at the root level.
	//
	// The following equation explains how each of the constants relate:
	// summaryL0Bits + (summaryLevels-1)*summaryLevelBits + logPallocChunkBytes = heapAddrBits
	//
	// summaryLevels is an architecture-dependent value defined in mpagealloc_*.go.
	summaryLevelBits = 3
	summaryL0Bits    = heapAddrBits - logPallocChunkBytes - (summaryLevels-1)*summaryLevelBits
)

const (
	// maxPackedValue is the maximum value that any of the three fields in
	// the pallocSum may take on.
	maxPackedValue    = 1 << logMaxPackedValue
	logMaxPackedValue = logPallocChunkPages + (summaryLevels-1)*summaryLevelBits
)

// pallocSum is a packed summary type which packs three numbers: start, max,
// and end into a single 8-byte value. Each of these values are a summary of
// a bitmap and are thus counts, each of which may have a maximum value of
// 2^21 - 1, or all three may be equal to 2^21. The latter case is represented
// by just setting the 64th bit.
type pallocSum uint64

// packPallocSum takes a start, max, and end value and produces a pallocSum.
func packPallocSum(start, max, end uint) pallocSum {
	if max == maxPackedValue {
		return pallocSum(uint64(1 << 63))
	}
	return pallocSum((uint64(start) & (maxPackedValue - 1)) |
		((uint64(max) & (maxPackedValue - 1)) << logMaxPackedValue) |
		((uint64(end) & (maxPackedValue - 1)) << (2 * logMaxPackedValue)))
}

// start extracts the start value from a packed sum.
func (p pallocSum) start() uint {
	if uint64(p)&uint64(1<<63) != 0 {
		return maxPackedValue
	}
	return uint(uint64(p) & (maxPackedValue - 1))
}

// max extracts the max value from a packed sum.
func (p pallocSum) max() uint {
	if uint64(p)&uint64(1<<63) != 0 {
		return maxPackedValue
	}
	return uint((uint64(p) >> logMaxPackedValue) & (maxPackedValue - 1))
}

// end extracts the end value from a packed sum.
func (p pallocSum) end() uint {
	if uint64(p)&uint64(1<<63) != 0 {
		return maxPackedValue
	}
	return uint((uint64(p) >> (2 * logMaxPackedValue)) & (maxPackedValue - 1))
}

// unpack unpacks all three values from the summary.
func (p pallocSum) unpack() (uint, uint, uint) {
	if uint64(p)&uint64(1<<63) != 0 {
		return maxPackedValue, maxPackedValue, maxPackedValue
	}
	return uint(uint64(p) & (maxPackedValue - 1)),
		uint((uint64(p) >> logMaxPackedValue) & (maxPackedValue - 1)),
		uint((uint64(p) >> (2 * logMaxPackedValue)) & (maxPackedValue - 1))
}
