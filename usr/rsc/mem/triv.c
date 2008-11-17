// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trivial base allocator.

#include "malloc.h"

// TODO: The call to sys·mmap should be a call to an assembly
// function sys·mmapnew that takes only a size parameter.
enum
{
	PROT_NONE	= 0x00,
	PROT_READ	= 0x01,
	PROT_WRITE	= 0x02,
	PROT_EXEC	= 0x04,

	MAP_FILE	= 0x0000,
	MAP_SHARED	= 0x0001,
	MAP_PRIVATE	= 0x0002,
	MAP_FIXED	= 0x0010,
	MAP_ANON	= 0x1000,
};

// Allocate and return zeroed memory.
// Simple allocator for small things like Span structures,
// and also used to grab large amounts of memory for
// the real allocator to hand out.
enum
{
	Round = 15,
};
void*
trivalloc(int32 size)
{
	static byte *p;
	static int32 n;
	byte *v;

	if(allocator·frozen)
		throw("allocator frozen");

//prints("Newmem: ");
//sys·printint(size);
//prints("\n");

	if(size < 4096) {	// TODO: Tune constant.
		size = (size + Round) & ~Round;
		if(size > n) {
			n = 1<<20;	// TODO: Tune constant.
			p = sys·mmap(nil, n, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
			allocator·footprint += n;
		}
		v = p;
		p += size;
		return v;
	}
	if(size & PageMask)
		size += (1<<PageShift) - (size & PageMask);
	v = sys·mmap(nil, size, PROT_READ|PROT_WRITE, MAP_ANON|MAP_PRIVATE, 0, 0);
	allocator·footprint += size;
	return v;
}

