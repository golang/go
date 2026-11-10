// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"internal/cpu"
	"unsafe"
)

// cacheLineSize is used to prevent false sharing of cache lines.
const cacheLineSize = max(unsafe.Sizeof(cpu.CacheLinePad{}), uintptr(64))

// counterBlockLen is the number of uint64 values a block contains.
// A block is sized to span exactly one cache line.
const counterBlockLen = int(cacheLineSize / unsafe.Sizeof(uint64(0)))

// counterBlock contains a collection of uint64 which fit into a cache line.
// It is intended to be accessed by a single P.
type counterBlock struct {
	line [counterBlockLen]uint64
}

// counterBlocks holds one counterBlock per P. Element pID holds the counters
// owned by P pID
type counterBlocks []counterBlock

var (
	countBlockMu      Mutex
	countBlocksActive counterBlocks
	countBlocksIndex  int
)

// allocCounters returns an iterator of pointers to uint64 values of the provided length. Each uint64
// is guaranteed to be located in a P-local cache line to minimize false sharing.
func allocCounters(size int) func(yield func(*uint64) bool) {
	countBlockMu.Lock()
	defer countBlockMu.Unlock()

	// A new counterBlocks is allocated when:
	//   - size > len(countBlocksActive): the requested number of Ps has grown
	//     (a nil countBlocksActive has length 0, so this also covers the
	//     not-yet-allocated case).
	//   - countBlocksIndex == counterBlockLen: every column of the current
	//     block-set has been handed out.
	if size > len(countBlocksActive) || countBlocksIndex == counterBlockLen {
		countBlocksActive = newCounterBlocks(size)
		countBlocksIndex = 0
	}
	iterCb := countBlocksActive
	iterCbIndex := countBlocksIndex
	countBlocksIndex++

	return func(yield func(*uint64) bool) {
		for pID := range size {
			c := &iterCb[pID].line[iterCbIndex]
			if !yield(c) {
				return
			}
		}
	}
}

// newCounterBlocks allocates size cache-line-aligned blocks, one per P.
//
// Because the runtime allocator guarantees only 8-byte pointer alignment for slice
// backing arrays, an unaligned block array could straddle cache lines and cause
// false sharing between adjacent Ps. To guarantee cache-line alignment,
// newCounterBlocks over-allocates by one block and reslices from the first
// cache line boundary.
func newCounterBlocks(size int) counterBlocks {
	// Allocate size+1 blocks so there is always at least one cache line aligned window of length size.
	buf := make(counterBlocks, size+1)
	// Compute the byte offset to the next cache line boundary.
	off := -uintptr(unsafe.Pointer(&buf[0])) & (cacheLineSize - 1)
	// Return a slice of length size starting at the cache line aligned address within buf.
	return unsafe.Slice((*counterBlock)(unsafe.Add(unsafe.Pointer(&buf[0]), off)), size)
}
