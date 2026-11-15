// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/runtime/atomic"
	"internal/runtime/sys"
	"unsafe"
)

type traceRegionAlloc struct {
	lock     mutex
	dropping atomic.Bool
	current  atomic.UnsafePointer
	full     *traceRegionAllocBlock
}

type traceRegionAllocBlock struct {
	_ sys.NotInHeap
	traceRegionAllocBlockHeader
	data [traceRegionAllocBlockData]byte
}

type traceRegionAllocBlockHeader struct {
	next *traceRegionAllocBlock
	off  atomic.Uintptr
}

const traceRegionAllocBlockData = 64<<10 - unsafe.Sizeof(traceRegionAllocBlockHeader{})

func (a *traceRegionAlloc) alloc(n uintptr) *notInHeap {
	n = alignUp(n, 8)
	if n > traceRegionAllocBlockData {
		throw("traceRegion: alloc too large")
	}
	if a.dropping.Load() {
		throw("traceRegion: alloc with concurrent drop")
	}

	block := (*traceRegionAllocBlock)(a.current.Load())
	if block != nil {
		r := block.off.Add(n)
		if r <= uintptr(len(block.data)) {
			return (*notInHeap)(unsafe.Pointer(&block.data[r-n]))
		}
	}

	var x *notInHeap
	systemstack(func() {
		lock(&a.lock)

		block = (*traceRegionAllocBlock)(a.current.Load())
		if block != nil {
			r := block.off.Add(n)
			if r <= uintptr(len(block.data)) {
				unlock(&a.lock)
				x = (*notInHeap)(unsafe.Pointer(&block.data[r-n]))
				return
			}

			block.next = a.full
			a.full = block
		}

		block = (*traceRegionAllocBlock)(sysAlloc(unsafe.Sizeof(traceRegionAllocBlock{}), &memstats.other_sys, "trace arena alloc"))
		if block == nil {
			throw("traceRegion: out of memory")
		}

		block.off.Store(n)
		x = (*notInHeap)(unsafe.Pointer(&block.data[0]))

		a.current.StoreNoWB(unsafe.Pointer(block))
		unlock(&a.lock)
	})
	return x
}

func (a *traceRegionAlloc) drop() {
	a.dropping.Store(true)
	for a.full != nil {
		block := a.full
		a.full = block.next
		sysFree(unsafe.Pointer(block), unsafe.Sizeof(traceRegionAllocBlock{}), &memstats.other_sys)
	}
	if current := a.current.Load(); current != nil {
		sysFree(current, unsafe.Sizeof(traceRegionAllocBlock{}), &memstats.other_sys)
		a.current.StoreNoWB(nil)
	}
	a.dropping.Store(false)
}
