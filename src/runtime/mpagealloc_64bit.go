// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || loong64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x

package runtime

import (
	"unsafe"
)

const (
	// The number of levels in the radix tree.
	summaryLevels = 5

	// Constants for testing.
	pageAlloc32Bit = 0
	pageAlloc64Bit = 1

	// Number of bits needed to represent all indices into the L1 of the
	// chunks map.
	//
	// See (*pageAlloc).chunks for more details. Update the documentation
	// there should this number change.
	pallocChunksL1Bits = 13
)

// levelBits is the number of bits in the radix for a given level in the super summary
// structure.
//
// The sum of all the entries of levelBits should equal heapAddrBits.
var levelBits = [summaryLevels]uint{
	summaryL0Bits,
	summaryLevelBits,
	summaryLevelBits,
	summaryLevelBits,
	summaryLevelBits,
}

// levelShift is the number of bits to shift to acquire the radix for a given level
// in the super summary structure.
//
// With levelShift, one can compute the index of the summary at level l related to a
// pointer p by doing:
//
//	p >> levelShift[l]
var levelShift = [summaryLevels]uint{
	heapAddrBits - summaryL0Bits,
	heapAddrBits - summaryL0Bits - 1*summaryLevelBits,
	heapAddrBits - summaryL0Bits - 2*summaryLevelBits,
	heapAddrBits - summaryL0Bits - 3*summaryLevelBits,
	heapAddrBits - summaryL0Bits - 4*summaryLevelBits,
}

// levelLogPages is log2 the maximum number of runtime pages in the address space
// a summary in the given level represents.
//
// The leaf level always represents exactly log2 of 1 chunk's worth of pages.
var levelLogPages = [summaryLevels]uint{
	logPallocChunkPages + 4*summaryLevelBits,
	logPallocChunkPages + 3*summaryLevelBits,
	logPallocChunkPages + 2*summaryLevelBits,
	logPallocChunkPages + 1*summaryLevelBits,
	logPallocChunkPages,
}

// sysInit performs architecture-dependent initialization of fields
// in pageAlloc. pageAlloc should be uninitialized except for sysStat
// if any runtime statistic should be updated.
func (p *pageAlloc) sysInit(test bool) {
	// Reserve memory for each level. This will get mapped in
	// as R/W by setArenas.
	for l, shift := range levelShift {
		entries := 1 << (heapAddrBits - shift)

		// Reserve b bytes of memory anywhere in the address space.
		b := alignUp(uintptr(entries)*pallocSumBytes, physPageSize)
		r := sysReserve(nil, b, "page summary")
		if r == nil {
			throw("failed to reserve page summary memory")
		}

		// Put this reservation into a slice.
		sl := notInHeapSlice{(*notInHeap)(r), 0, entries}
		p.summary[l] = *(*[]pallocSum)(unsafe.Pointer(&sl))
	}
}

// sysGrow performs architecture-dependent operations on heap
// growth for the page allocator, such as mapping in new memory
// for summaries. It also updates the length of the slices in
// p.summary.
//
// base is the base of the newly-added heap memory and limit is
// the first address past the end of the newly-added heap memory.
// Both must be aligned to pallocChunkBytes.
//
// The caller must update p.start and p.end after calling sysGrow.
func (p *pageAlloc) sysGrow(base, limit uintptr) {
	if base%pallocChunkBytes != 0 || limit%pallocChunkBytes != 0 {
		print("runtime: base = ", hex(base), ", limit = ", hex(limit), "\n")
		throw("sysGrow bounds not aligned to pallocChunkBytes")
	}

	// addrRangeToSummaryRange converts a range of addresses into a range
	// of summary indices which must be mapped to support those addresses
	// in the summary range.
	addrRangeToSummaryRange := func(level int, r addrRange) (int, int) {
		sumIdxBase, sumIdxLimit := addrsToSummaryRange(level, r.base.addr(), r.limit.addr())
		return blockAlignSummaryRange(level, sumIdxBase, sumIdxLimit)
	}

	// summaryRangeToSumAddrRange converts a range of indices in any
	// level of p.summary into page-aligned addresses which cover that
	// range of indices.
	summaryRangeToSumAddrRange := func(level, sumIdxBase, sumIdxLimit int) addrRange {
		baseOffset := alignDown(uintptr(sumIdxBase)*pallocSumBytes, physPageSize)
		limitOffset := alignUp(uintptr(sumIdxLimit)*pallocSumBytes, physPageSize)
		base := unsafe.Pointer(&p.summary[level][0])
		return addrRange{
			offAddr{uintptr(add(base, baseOffset))},
			offAddr{uintptr(add(base, limitOffset))},
		}
	}

	// addrRangeToSumAddrRange is a convenience function that converts
	// an address range r to the address range of the given summary level
	// that stores the summaries for r.
	addrRangeToSumAddrRange := func(level int, r addrRange) addrRange {
		sumIdxBase, sumIdxLimit := addrRangeToSummaryRange(level, r)
		return summaryRangeToSumAddrRange(level, sumIdxBase, sumIdxLimit)
	}

	// Find the first inUse index which is strictly greater than base.
	//
	// Because this function will never be asked remap the same memory
	// twice, this index is effectively the index at which we would insert
	// this new growth, and base will never overlap/be contained within
	// any existing range.
	//
	// This will be used to look at what memory in the summary array is already
	// mapped before and after this new range.
	inUseIndex := p.inUse.findSucc(base)

	// Walk up the radix tree and map summaries in as needed.
	for l := range p.summary {
		// Figure out what part of the summary array this new address space needs.
		needIdxBase, needIdxLimit := addrRangeToSummaryRange(l, makeAddrRange(base, limit))

		// Update the summary slices with a new upper-bound. This ensures
		// we get tight bounds checks on at least the top bound.
		//
		// We must do this regardless of whether we map new memory.
		if needIdxLimit > len(p.summary[l]) {
			p.summary[l] = p.summary[l][:needIdxLimit]
		}

		// Compute the needed address range in the summary array for level l.
		need := summaryRangeToSumAddrRange(l, needIdxBase, needIdxLimit)

		// Prune need down to what needs to be newly mapped. Some parts of it may
		// already be mapped by what inUse describes due to page alignment requirements
		// for mapping. Because this function will never be asked to remap the same
		// memory twice, it should never be possible to prune in such a way that causes
		// need to be split.
		if inUseIndex > 0 {
			need = need.subtract(addrRangeToSumAddrRange(l, p.inUse.ranges[inUseIndex-1]))
		}
		if inUseIndex < len(p.inUse.ranges) {
			need = need.subtract(addrRangeToSumAddrRange(l, p.inUse.ranges[inUseIndex]))
		}
		// It's possible that after our pruning above, there's nothing new to map.
		if need.size() == 0 {
			continue
		}

		// Map and commit need.
		sysMap(unsafe.Pointer(need.base.addr()), need.size(), p.sysStat, "page alloc")
		sysUsed(unsafe.Pointer(need.base.addr()), need.size(), need.size())
		p.summaryMappedReady += need.size()
	}
}

// sysGrow increases the index's backing store in response to a heap growth.
//
// Returns the amount of memory added to sysStat.
func (s *scavengeIndex) sysGrow(base, limit uintptr, sysStat *sysMemStat) uintptr {
	if base%pallocChunkBytes != 0 || limit%pallocChunkBytes != 0 {
		print("runtime: base = ", hex(base), ", limit = ", hex(limit), "\n")
		throw("sysGrow bounds not aligned to pallocChunkBytes")
	}
	scSize := unsafe.Sizeof(atomicScavChunkData{})
	// Map and commit the pieces of chunks that we need.
	//
	// We always map the full range of the minimum heap address to the
	// maximum heap address. We don't do this for the summary structure
	// because it's quite large and a discontiguous heap could cause a
	// lot of memory to be used. In this situation, the worst case overhead
	// is in the single-digit MiB if we map the whole thing.
	//
	// The base address of the backing store is always page-aligned,
	// because it comes from the OS, so it's sufficient to align the
	// index.
	haveMin := s.min.Load()
	haveMax := s.max.Load()
	needMin := alignDown(uintptr(chunkIndex(base)), physPageSize/scSize)
	needMax := alignUp(uintptr(chunkIndex(limit)), physPageSize/scSize)

	// We need a contiguous range, so extend the range if there's no overlap.
	if needMax < haveMin {
		needMax = haveMin
	}
	if haveMax != 0 && needMin > haveMax {
		needMin = haveMax
	}

	// Avoid a panic from indexing one past the last element.
	chunksBase := uintptr(unsafe.Pointer(&s.chunks[0]))
	have := makeAddrRange(chunksBase+haveMin*scSize, chunksBase+haveMax*scSize)
	need := makeAddrRange(chunksBase+needMin*scSize, chunksBase+needMax*scSize)

	// Subtract any overlap from rounding. We can't re-map memory because
	// it'll be zeroed.
	need = need.subtract(have)

	// If we've got something to map, map it, and update the slice bounds.
	if need.size() != 0 {
		sysMap(unsafe.Pointer(need.base.addr()), need.size(), sysStat, "scavenge index")
		sysUsed(unsafe.Pointer(need.base.addr()), need.size(), need.size())
		// Update the indices only after the new memory is valid.
		if haveMax == 0 || needMin < haveMin {
			s.min.Store(needMin)
		}
		if needMax > haveMax {
			s.max.Store(needMax)
		}
	}
	return need.size()
}

// sysInit initializes the scavengeIndex' chunks array.
//
// Returns the amount of memory added to sysStat.
func (s *scavengeIndex) sysInit(test bool, sysStat *sysMemStat) uintptr {
	n := uintptr(1<<heapAddrBits) / pallocChunkBytes
	nbytes := n * unsafe.Sizeof(atomicScavChunkData{})
	r := sysReserve(nil, nbytes, "scavenge index")
	sl := notInHeapSlice{(*notInHeap)(r), int(n), int(n)}
	s.chunks = *(*[]atomicScavChunkData)(unsafe.Pointer(&sl))
	return 0 // All memory above is mapped Reserved.
}
