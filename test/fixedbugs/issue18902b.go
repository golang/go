// skip

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import (
	"unsafe"
)

type gcMaxTreeNodeVal uint64

var work struct {
	full         uint64    // lock-free list of full blocks workbuf
	empty        uint64    // lock-free list of empty blocks workbuf
	pad0         [64]uint8 // prevents false-sharing between full/empty and nproc/nwait
	bytesMarked  uint64
	markrootNext uint32 // next markroot job
	markrootJobs uint32 // number of markroot jobs
	nproc        uint32
	tstart       int64
	nwait        uint32
	ndone        uint32
}

type gcShardQueue1 struct {
	partial *workbuf
	full    *workbuf
	n       uintptr
	maxTree gcMaxTreeNodeVal
}
type gcShardQueue struct {
	gcShardQueue1
	pad [64 - unsafe.Sizeof(gcShardQueue1{})]byte
}

const gcSortBufPointers = (64 << 10) / 8

type gcSortBuf struct {
	buf *gcSortArray
	tmp *gcSortArray
	n   uintptr
}

//go:notinheap
type gcSortArray [gcSortBufPointers]uintptr

const (
	_DebugGC             = 0
	_ConcurrentSweep     = true
	_FinBlockSize        = 4 * 1024
	sweepMinHeapDistance = 1024 * 1024
	gcShardShift         = 2 + 20
	gcShardBytes         = 1 << gcShardShift
)

//go:notinheap
type mheap struct {
	shardQueues       []gcShardQueue
	_                 uint32     // align uint64 fields on 32-bit for atomics
	pagesInUse        uint64     // pages of spans in stats _MSpanInUse; R/W with mheap.lock
	spanBytesAlloc    uint64     // bytes of spans allocated this cycle; updated atomically
	pagesSwept        uint64     // pages swept this cycle; updated atomically
	sweepPagesPerByte float64    // proportional sweep ratio; written with lock, read without
	largefree         uint64     // bytes freed for large objects (>maxsmallsize)
	nlargefree        uint64     // number of frees for large objects (>maxsmallsize)
	nsmallfree        [67]uint64 // number of frees for small objects (<=maxsmallsize)
	bitmap            uintptr    // Points to one byte past the end of the bitmap
	bitmap_mapped     uintptr
	arena_start       uintptr
	arena_used        uintptr // always mHeap_Map{Bits,Spans} before updating
	arena_end         uintptr
	arena_reserved    bool
}

var mheap_ mheap

type lfnode struct {
	next    uint64
	pushcnt uintptr
}
type workbufhdr struct {
	node lfnode // must be first
	next *workbuf
	nobj int
}

//go:notinheap
type workbuf struct {
	workbufhdr
	obj [(2048 - unsafe.Sizeof(workbufhdr{})) / 8]uintptr
}

//go:noinline
func (b *workbuf) checkempty() {
	if b.nobj != 0 {
		b.nobj = 0
	}
}
func putempty(b *workbuf) {
	b.checkempty()
	lfstackpush(&work.empty, &b.node)
}

//go:noinline
func lfstackpush(head *uint64, node *lfnode) {
}

//go:noinline
func (q *gcShardQueue) add(qidx uintptr, ptrs []uintptr, spare *workbuf) *workbuf {
	return spare
}

func (b *gcSortBuf) flush() {
	if b.n == 0 {
		return
	}
	const sortDigitBits = 11
	buf, tmp := b.buf[:b.n], b.tmp[:b.n]
	moreBits := true
	for shift := uint(gcShardShift); moreBits; shift += sortDigitBits {
		const k = 1 << sortDigitBits
		var pos [k]uint16
		nshift := shift + sortDigitBits
		nbits := buf[0] >> nshift
		moreBits = false
		for _, v := range buf {
			pos[(v>>shift)%k]++
			moreBits = moreBits || v>>nshift != nbits
		}
		var sum uint16
		for i, count := range &pos {
			pos[i] = sum
			sum += count
		}
		for _, v := range buf {
			digit := (v >> shift) % k
			tmp[pos[digit]] = v
			pos[digit]++
		}
		buf, tmp = tmp, buf
	}
	start := mheap_.arena_start
	i0 := 0
	shard0 := (buf[0] - start) / gcShardBytes
	var spare *workbuf
	for i, p := range buf {
		shard := (p - start) / gcShardBytes
		if shard != shard0 {
			spare = mheap_.shardQueues[shard0].add(shard0, buf[i0:i], spare)
			i0, shard0 = i, shard
		}
	}
	spare = mheap_.shardQueues[shard0].add(shard0, buf[i0:], spare)
	b.n = 0
	if spare != nil {
		putempty(spare)
	}
}
