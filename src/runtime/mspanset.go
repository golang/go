// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/cpu"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// A spanSet is a set of *mspans.
//
// spanSet is safe for concurrent push operations *or* concurrent
// pop operations, but not both simultaneously.
type spanSet struct {
	// A spanSet is a two-level data structure consisting of a
	// growable spine that points to fixed-sized blocks. The spine
	// can be accessed without locks, but adding a block or
	// growing it requires taking the spine lock.
	//
	// Because each mspan covers at least 8K of heap and takes at
	// most 8 bytes in the spanSet, the growth of the spine is
	// quite limited.
	//
	// The spine and all blocks are allocated off-heap, which
	// allows this to be used in the memory manager and avoids the
	// need for write barriers on all of these. We never release
	// this memory because there could be concurrent lock-free
	// access and we're likely to reuse it anyway. (In principle,
	// we could do this during STW.)

	spineLock mutex
	spine     unsafe.Pointer // *[N]*spanSetBlock, accessed atomically
	spineLen  uintptr        // Spine array length, accessed atomically
	spineCap  uintptr        // Spine array cap, accessed under lock

	// index is the first unused slot in the logical concatenation
	// of all blocks. It is accessed atomically.
	index uint32
}

const (
	spanSetBlockEntries = 512 // 4KB on 64-bit
	spanSetInitSpineCap = 256 // Enough for 1GB heap on 64-bit
)

type spanSetBlock struct {
	spans [spanSetBlockEntries]*mspan
}

// push adds span s to buffer b. push is safe to call concurrently
// with other push operations, but NOT to call concurrently with pop.
func (b *spanSet) push(s *mspan) {
	// Obtain our slot.
	cursor := uintptr(atomic.Xadd(&b.index, +1) - 1)
	top, bottom := cursor/spanSetBlockEntries, cursor%spanSetBlockEntries

	// Do we need to add a block?
	spineLen := atomic.Loaduintptr(&b.spineLen)
	var block *spanSetBlock
retry:
	if top < spineLen {
		spine := atomic.Loadp(unsafe.Pointer(&b.spine))
		blockp := add(spine, sys.PtrSize*top)
		block = (*spanSetBlock)(atomic.Loadp(blockp))
	} else {
		// Add a new block to the spine, potentially growing
		// the spine.
		lock(&b.spineLock)
		// spineLen cannot change until we release the lock,
		// but may have changed while we were waiting.
		spineLen = atomic.Loaduintptr(&b.spineLen)
		if top < spineLen {
			unlock(&b.spineLock)
			goto retry
		}

		if spineLen == b.spineCap {
			// Grow the spine.
			newCap := b.spineCap * 2
			if newCap == 0 {
				newCap = spanSetInitSpineCap
			}
			newSpine := persistentalloc(newCap*sys.PtrSize, cpu.CacheLineSize, &memstats.gc_sys)
			if b.spineCap != 0 {
				// Blocks are allocated off-heap, so
				// no write barriers.
				memmove(newSpine, b.spine, b.spineCap*sys.PtrSize)
			}
			// Spine is allocated off-heap, so no write barrier.
			atomic.StorepNoWB(unsafe.Pointer(&b.spine), newSpine)
			b.spineCap = newCap
			// We can't immediately free the old spine
			// since a concurrent push with a lower index
			// could still be reading from it. We let it
			// leak because even a 1TB heap would waste
			// less than 2MB of memory on old spines. If
			// this is a problem, we could free old spines
			// during STW.
		}

		// Allocate a new block and add it to the spine.
		block = (*spanSetBlock)(persistentalloc(unsafe.Sizeof(spanSetBlock{}), cpu.CacheLineSize, &memstats.gc_sys))
		blockp := add(b.spine, sys.PtrSize*top)
		// Blocks are allocated off-heap, so no write barrier.
		atomic.StorepNoWB(blockp, unsafe.Pointer(block))
		atomic.Storeuintptr(&b.spineLen, spineLen+1)
		unlock(&b.spineLock)
	}

	// We have a block. Insert the span atomically, since there may be
	// concurrent readers via the block API.
	atomic.StorepNoWB(unsafe.Pointer(&block.spans[bottom]), unsafe.Pointer(s))
}

// pop removes and returns a span from buffer b, or nil if b is empty.
// pop is safe to call concurrently with other pop operations, but NOT
// to call concurrently with push.
func (b *spanSet) pop() *mspan {
	cursor := atomic.Xadd(&b.index, -1)
	if int32(cursor) < 0 {
		atomic.Xadd(&b.index, +1)
		return nil
	}

	// There are no concurrent spine or block modifications during
	// pop, so we can omit the atomics.
	top, bottom := cursor/spanSetBlockEntries, cursor%spanSetBlockEntries
	blockp := (**spanSetBlock)(add(b.spine, sys.PtrSize*uintptr(top)))
	block := *blockp
	s := block.spans[bottom]
	// Clear the pointer for block(i).
	block.spans[bottom] = nil
	return s
}

// numBlocks returns the number of blocks in buffer b. numBlocks is
// safe to call concurrently with any other operation. Spans that have
// been pushed prior to the call to numBlocks are guaranteed to appear
// in some block in the range [0, numBlocks()), assuming there are no
// intervening pops. Spans that are pushed after the call may also
// appear in these blocks.
func (b *spanSet) numBlocks() int {
	return int((atomic.Load(&b.index) + spanSetBlockEntries - 1) / spanSetBlockEntries)
}

// block returns the spans in the i'th block of buffer b. block is
// safe to call concurrently with push. The block may contain nil
// pointers that must be ignored, and each entry in the block must be
// loaded atomically.
func (b *spanSet) block(i int) []*mspan {
	// Perform bounds check before loading spine address since
	// push ensures the allocated length is at least spineLen.
	if i < 0 || uintptr(i) >= atomic.Loaduintptr(&b.spineLen) {
		throw("block index out of range")
	}

	// Get block i.
	spine := atomic.Loadp(unsafe.Pointer(&b.spine))
	blockp := add(spine, sys.PtrSize*uintptr(i))
	block := (*spanSetBlock)(atomic.Loadp(blockp))

	// Slice the block if necessary.
	cursor := uintptr(atomic.Load(&b.index))
	top, bottom := cursor/spanSetBlockEntries, cursor%spanSetBlockEntries
	var spans []*mspan
	if uintptr(i) < top {
		spans = block.spans[:]
	} else {
		spans = block.spans[:bottom]
	}
	return spans
}
