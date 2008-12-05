// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../src/runtime/runtime.h"

typedef struct PageMap PageMap;

enum
{
	PageShift = 12,
	PageMask = (1<<PageShift) - 1,
};

#define RefFree	0xffffffffU
#define RefManual	0xfffffffeU
#define RefStack	0xfffffffdU

enum {
	PMBits = 64 - PageShift,
	PMLevels = 4,
	PMLevelBits = 13,
	PMLevelSize = 1<<PMLevelBits,
	PMLevelMask = PMLevelSize - 1,
};
struct PageMap
{
	void *level0[PMLevelSize];
};

typedef struct Span Span;
typedef struct Central Central;

// A Span contains metadata about a range of pages.
enum {
	SpanInUse = 0,	// span has been handed out by allocator
	SpanFree = 1,	// span is in central free list
};
struct Span
{
	Span *aprev;	// in list of all spans
	Span *anext;

	Span *next;	// in free lists
	byte *base;	// first byte in span
	uintptr length;	// number of pages in span
	int32 cl;
	int32 state;	// state (enum above)
	union {
		int32 ref;	// reference count if state == SpanInUse (for GC)
		int32 *refbase;	// ptr to packed ref counts
	};
//	void *type;	// object type if state == SpanInUse (for GC)
};

// The Central cache contains a list of free spans,
// as well as free lists of small blocks.
struct Central
{
	Lock;
	Span *free[256];
	Span *large;	// free spans >= MaxPage pages
};

extern int64 allocator·allocated;
extern int64 allocator·footprint;
extern bool allocator·frozen;

void* trivalloc(int32);
void* pmlookup(PageMap*, uintptr);
void* pminsert(PageMap*, uintptr, void*);

void*	alloc(int32);
void	free(void*);
bool	findobj(void*, void**, int64*, int32**);

extern Central central;
extern PageMap spanmap;
extern int32 classtosize[SmallFreeClasses];
extern Span *spanfirst, *spanlast;
