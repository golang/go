// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goexperiment"
	"internal/runtime/gc"
	"unsafe"
)

// tryMallocBatch32 allocates adjacent but separately collectible objects from
// already-free slots in one cached span. The caller falls back to separate
// allocations if this operation would need any allocator slow path.
func tryMallocBatch32(out []*[4]uint64) bool {
	n := len(out)
	if n < 2 || n > 8 || runtimeFreegcEnabled || !sizeSpecializedMallocEnabled {
		return false
	}

	const size = unsafe.Sizeof([4]uint64{})
	spc := makeSpanClass(gc.SizeToSizeClass8[size/gc.SmallSizeDiv], true)
	bytes := uintptr(n) * size

	mp := acquirem()
	if debug.malloc || gcBlackenEnabled != 0 || writeBarrier.enabled || (goexperiment.RuntimeSecret && getg().secret > 0) {
		releasem(mp)
		return false
	}
	c := getMCache(mp)
	if c.memProfRate != MemProfileRate || c.nextSample < int64(bytes) {
		releasem(mp)
		return false
	}

	span := c.alloc[spc]
	start := span.freeindex
	end := start + uint16(n)
	mask := (uint64(1) << uint(n)) - 1
	if end > span.nelems || (end != span.nelems && (start%64+uint16(n)) >= 64) || span.allocCache&mask != mask {
		releasem(mp)
		return false
	}

	mp.mallocing = 1
	span.freeindex = end
	span.allocCache >>= uint(n)
	span.allocCount += uint16(n)
	first := unsafe.Pointer(span.base() + uintptr(start)*size)
	if span.needzero != 0 {
		memclrNoHeapPointers(first, bytes)
	}

	publicationBarrier()
	span.freeIndexForScan = end
	c.nextSample -= int64(bytes)
	for i := range out {
		out[i] = (*[4]uint64)(add(first, uintptr(i)*size))
	}

	mp.mallocing = 0
	releasem(mp)
	return true
}

// allocSlice32 fills a slice of pointers to 32-byte objects that contain no
// pointers. The compiler calls it only for a loop that assigns new(T) to each
// element of the same slice.
func allocSlice32(p unsafe.Pointer, n int, typ *_type) {
	dst := unsafe.Slice((**[4]uint64)(p), n)
	const batchSize = 8
	for sizeSpecializedMallocEnabled && len(dst) >= batchSize {
		batch := dst[:batchSize]
		if !tryMallocBatch32(batch) {
			for i := range batch {
				batch[i] = (*[4]uint64)(mallocgcSmallNoScanSC4(32, typ, true))
			}
		}
		dst = dst[batchSize:]
	}
	for i := range dst {
		if sizeSpecializedMallocEnabled {
			dst[i] = (*[4]uint64)(mallocgcSmallNoScanSC4(32, typ, true))
		} else {
			dst[i] = (*[4]uint64)(mallocgc(32, typ, true))
		}
	}
}
