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

extern int64 allocator·allocated;
extern int64 allocator·footprint;
extern bool allocator·frozen;

void* trivalloc(int32);
void* pmlookup(PageMap*, uintptr);
void* pminsert(PageMap*, uintptr, void*);

void*	alloc(int32);
void	free(void*);
