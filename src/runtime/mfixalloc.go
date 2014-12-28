// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fixed-size object allocator.  Returned memory is not zeroed.
//
// See malloc.h for overview.

package runtime

import "unsafe"

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
