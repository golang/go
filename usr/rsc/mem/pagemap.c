// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "malloc.h"

// A PageMap maps page numbers to void* pointers.
// The AMD64 has 64-bit addresses and 4096-byte pages, so
// the page numbers are 52 bits.  We use a four-level radix tree,
// with 13 bits for each level.  This requires 32 kB per level or
// 128 kB for a table with one entry.  Moving to three levels of 18 bits
// would require 3 MB for a table with one entry, which seems too expensive.
// This is easy to change.
// It may be that a balanced tree would be better anyway.

// Return the entry for page number pn in m.
void*
pmlookup(PageMap *m, uintptr pn)
{
	int32 i, x;
	void **v;

	v = &m->level0[0];
	for(i=0; i<PMLevels; i++) {
		// Pick off top PMLevelBits bits as index and shift up.
		x = (pn >> (PMBits - PMLevelBits)) & PMLevelMask;
		pn <<= PMLevelBits;

		// Walk down using index.
		v = v[x];
		if(v == nil)
			return nil;
	}
	return v;
}

// Set the entry for page number pn in m to s.
// Return the old value.
void*
pminsert(PageMap *m, uintptr pn, void *value)
{
	int32 i, x;
	void **v, **l;

	l = nil;	// shut up 6c
	v = &m->level0[0];
	for(i=0; i<PMLevels; i++) {
		// Pick off top PMLevelBits bits as index and shift up.
		x = (pn >> (PMBits - PMLevelBits)) & PMLevelMask;
		pn <<= PMLevelBits;

		// Walk down using index, but remember location of pointer.
		l = &v[x];
		v = *l;

		// Allocate new level if needed.
		if(v == nil && i < PMLevels-1) {
			v = trivalloc(PMLevelSize * sizeof v[0]);
			*l = v;
		}
	}

	// Record new value and return old.
	*l = value;
	return v;
}
