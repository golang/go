// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fixed-size object allocator.  Returned memory is not zeroed.
//
// See malloc.go for overview.

package runtime

import "unsafe"

// FixAlloc is a simple free-list allocator for fixed size objects.
// Malloc uses a FixAlloc wrapped around sysAlloc to manages its
// MCache and MSpan objects.
//
// Memory returned by FixAlloc_Alloc is not zeroed.
// The caller is responsible for locking around FixAlloc calls.
// Callers can keep state in the object but the first word is
// smashed by freeing and reallocating.
type fixalloc struct {
	size   uintptr
	first  unsafe.Pointer // go func(unsafe.pointer, unsafe.pointer); f(arg, p) called first time p is returned
	arg    unsafe.Pointer
	list   *mlink
	chunk  *byte
	nchunk uint32
	inuse  uintptr // in-use bytes now
	stat   *uint64
}

// A generic linked list of blocks.  (Typically the block is bigger than sizeof(MLink).)
// Since assignments to mlink.next will result in a write barrier being preformed
// this can not be used by some of the internal GC structures. For example when
// the sweeper is placing an unmarked object on the free list it does not want the
// write barrier to be called since that could result in the object being reachable.
type mlink struct {
	next *mlink
}

// Initialize f to allocate objects of the given size,
// using the allocator to obtain chunks of memory.
func fixAlloc_Init(f *fixalloc, size uintptr, first func(unsafe.Pointer, unsafe.Pointer), arg unsafe.Pointer, stat *uint64) {
	f.size = size
	f.first = *(*unsafe.Pointer)(unsafe.Pointer(&first))
	f.arg = arg
	f.list = nil
	f.chunk = nil
	f.nchunk = 0
	f.inuse = 0
	f.stat = stat
}

func fixAlloc_Alloc(f *fixalloc) unsafe.Pointer {
	if f.size == 0 {
		print("runtime: use of FixAlloc_Alloc before FixAlloc_Init\n")
		throw("runtime: internal error")
	}

	if f.list != nil {
		v := unsafe.Pointer(f.list)
		f.list = f.list.next
		f.inuse += f.size
		return v
	}
	if uintptr(f.nchunk) < f.size {
		f.chunk = (*uint8)(persistentalloc(_FixAllocChunk, 0, f.stat))
		f.nchunk = _FixAllocChunk
	}

	v := (unsafe.Pointer)(f.chunk)
	if f.first != nil {
		fn := *(*func(unsafe.Pointer, unsafe.Pointer))(unsafe.Pointer(&f.first))
		fn(f.arg, v)
	}
	f.chunk = (*byte)(add(unsafe.Pointer(f.chunk), f.size))
	f.nchunk -= uint32(f.size)
	f.inuse += f.size
	return v
}

func fixAlloc_Free(f *fixalloc, p unsafe.Pointer) {
	f.inuse -= f.size
	v := (*mlink)(p)
	v.next = f.list
	f.list = v
}
