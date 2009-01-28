// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fixed-size object allocator.  Returned memory is not zeroed.
//
// See malloc.h for overview.

#include "runtime.h"
#include "malloc.h"

// Initialize f to allocate objects of the given size,
// using the allocator to obtain chunks of memory.
void
FixAlloc_Init(FixAlloc *f, uintptr size, void *(*alloc)(uintptr), void (*first)(void*, byte*), void *arg)
{
	f->size = size;
	f->alloc = alloc;
	f->first = first;
	f->arg = arg;
	f->list = nil;
	f->chunk = nil;
	f->nchunk = 0;
}

void*
FixAlloc_Alloc(FixAlloc *f)
{
	void *v;

	if(f->list) {
		v = f->list;
		f->list = *(void**)f->list;
		return v;
	}
	if(f->nchunk < f->size) {
		f->chunk = f->alloc(FixAllocChunk);
		if(f->chunk == nil)
			throw("out of memory (FixAlloc)");
		f->nchunk = FixAllocChunk;
	}
	v = f->chunk;
	if(f->first)
		f->first(f->arg, v);
	f->chunk += f->size;
	f->nchunk -= f->size;
	return v;
}

void
FixAlloc_Free(FixAlloc *f, void *p)
{
	*(void**)p = f->list;
	f->list = p;
}

