// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package calloc

// This package contains a simple "batch" allocator for allocating
// coverage counters (slices of uint32 basically), for working with
// coverage data files. Collections of counter arrays tend to all be
// live/dead over the same time period, so a good fit for batch
// allocation.

type BatchCounterAlloc struct {
	pool []uint32
}

func (ca *BatchCounterAlloc) AllocateCounters(n int) []uint32 {
	const chunk = 8192
	if n > cap(ca.pool) {
		siz := chunk
		if n > chunk {
			siz = n
		}
		ca.pool = make([]uint32, siz)
	}
	rv := ca.pool[:n]
	ca.pool = ca.pool[n:]
	return rv
}
