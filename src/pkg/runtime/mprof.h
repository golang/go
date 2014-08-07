// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Per-call-stack profiling information.
// Lookup by hashing call stack into a linked-list hash table.
struct Bucket
{
	Bucket	*next;	// next in hash list
	Bucket	*allnext;	// next in list of all mbuckets/bbuckets
	int32	typ;
	// Generally unions can break precise GC,
	// this one is fine because it does not contain pointers.
	union
	{
		struct  // typ == MProf
		{
			// The following complex 3-stage scheme of stats accumulation
			// is required to obtain a consistent picture of mallocs and frees
			// for some point in time.
			// The problem is that mallocs come in real time, while frees
			// come only after a GC during concurrent sweeping. So if we would
			// naively count them, we would get a skew toward mallocs.
			//
			// Mallocs are accounted in recent stats.
			// Explicit frees are accounted in recent stats.
			// GC frees are accounted in prev stats.
			// After GC prev stats are added to final stats and
			// recent stats are moved into prev stats.
			uintptr	allocs;
			uintptr	frees;
			uintptr	alloc_bytes;
			uintptr	free_bytes;

			uintptr	prev_allocs;  // since last but one till last gc
			uintptr	prev_frees;
			uintptr	prev_alloc_bytes;
			uintptr	prev_free_bytes;

			uintptr	recent_allocs;  // since last gc till now
			uintptr	recent_frees;
			uintptr	recent_alloc_bytes;
			uintptr	recent_free_bytes;

		} mp;
		struct  // typ == BProf
		{
			int64	count;
			int64	cycles;
		} bp;
	} data;
	uintptr	hash;	// hash of size + stk
	uintptr	size;
	uintptr	nstk;
	uintptr	stk[1];
};
