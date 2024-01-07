// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.exectracer2

// Simple not-in-heap bump-pointer traceRegion allocator.

package runtime

import (
	"internal/goarch"
	"runtime/internal/sys"
	"unsafe"
)

// traceRegionAlloc is a non-thread-safe region allocator.
// It holds a linked list of traceRegionAllocBlock.
type traceRegionAlloc struct {
	head *traceRegionAllocBlock
	off  uintptr
}

// traceRegionAllocBlock is a block in traceRegionAlloc.
//
// traceRegionAllocBlock is allocated from non-GC'd memory, so it must not
// contain heap pointers. Writes to pointers to traceRegionAllocBlocks do
// not need write barriers.
type traceRegionAllocBlock struct {
	_    sys.NotInHeap
	next *traceRegionAllocBlock
	data [64<<10 - goarch.PtrSize]byte
}

// alloc allocates n-byte block.
func (a *traceRegionAlloc) alloc(n uintptr) *notInHeap {
	n = alignUp(n, goarch.PtrSize)
	if a.head == nil || a.off+n > uintptr(len(a.head.data)) {
		if n > uintptr(len(a.head.data)) {
			throw("traceRegion: alloc too large")
		}
		block := (*traceRegionAllocBlock)(sysAlloc(unsafe.Sizeof(traceRegionAllocBlock{}), &memstats.other_sys))
		if block == nil {
			throw("traceRegion: out of memory")
		}
		block.next = a.head
		a.head = block
		a.off = 0
	}
	p := &a.head.data[a.off]
	a.off += n
	return (*notInHeap)(unsafe.Pointer(p))
}

// drop frees all previously allocated memory and resets the allocator.
func (a *traceRegionAlloc) drop() {
	for a.head != nil {
		block := a.head
		a.head = block.next
		sysFree(unsafe.Pointer(block), unsafe.Sizeof(traceRegionAllocBlock{}), &memstats.other_sys)
	}
}
