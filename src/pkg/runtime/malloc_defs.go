// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go definitions of internal structures. Master is malloc.h

package runtime

import "unsafe"

const (
	pageShift = 12
	pageSize  = 1 << pageShift
	pageMask  = pageSize - 1
)

type pageID uintptr

const (
	numSizeClasses   = 67
	maxSmallSize     = 32 << 10
	fixAllocChunk    = 128 << 10
	maxMCacheListLen = 256
	maxMCacheSize    = 2 << 20
	maxMHeapList     = 1 << 8 // 1 << (20 - pageShift)
	heapAllocChunk   = 1 << 20
)

type mLink struct {
	next *mLink
}

type fixAlloc struct {
	size   uintptr
	alloc  func(uintptr)
	first  func(unsafe.Pointer, *byte)
	arg    unsafe.Pointer
	list   *mLink
	chunk  *byte
	nchunk uint32
	inuse  uintptr
	sys    uintptr
}


// MStats? used to be in extern.go

type mCacheList struct {
	list     *mLink
	nlist    uint32
	nlistmin uint32
}

type mCache struct {
	list          [numSizeClasses]mCacheList
	size          uint64
	local_alloc   int64
	local_objects int64
	next_sample   int32
}

type mSpan struct {
	next      *mSpan
	prev      *mSpan
	allnext   *mSpan
	start     pageID
	npages    uintptr
	freelist  *mLink
	ref       uint32
	sizeclass uint32
	state     uint32
	//	union {
	gcref *uint32 // sizeclass > 0
	//		gcref0 uint32;	// sizeclass == 0
	//	}
}

type mCentral struct {
	lock
	sizeclass int32
	nonempty  mSpan
	empty     mSpan
	nfree     int32
}

type mHeap struct {
	lock
	free        [maxMHeapList]mSpan
	large       mSpan
	allspans    *mSpan
	min         *byte
	max         *byte
	closure_min *byte
	closure_max *byte

	central [numSizeClasses]struct {
		pad [64]byte
		// union: mCentral
	}

	spanalloc  fixAlloc
	cachealloc fixAlloc
}

const (
	refFree = iota
	refStack
	refNone
	refSome
	refcountOverhead = 4
	refNoPointers    = 0x80000000
	refHasFinalizer  = 0x40000000
	refProfiled      = 0x20000000
	refNoProfiling   = 0x10000000
	refFlags         = 0xFFFF0000
)

const (
	mProf_None = iota
	mProf_Sample
	mProf_All
)

type finalizer struct {
	next *finalizer
	fn   func(unsafe.Pointer)
	arg  unsafe.Pointer
	nret int32
}
