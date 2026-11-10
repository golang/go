// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"runtime"
	"sync/atomic"
	"unsafe"
)

// stripeArr is the slice of pointers to uint64. Each index in the slice
// corresponds to a uint64 which should only be modified by a single P. Each
// uint64 is stored in a counterBlock of memory that is associated with a unique P.
// The stripe block contains values for a counter that span multiple different counterBlocks.
type stripeArr []*uint64

// stripe handles the dynamic growth of the sharded counters.
// It can only grow.
//
// Synchronization protocol:
//   - mu serializes growth only (grow, and the check-then-grow in getSlow).
//     Readers on the fast path never take mu.
//   - block and stripeCap are published atomically. grow stores block before
//     stripeCap; readers (get/getSlow/Value) load stripeCap before block.
//     Because block only ever grows and always preserves its existing prefix,
//     a reader may observe a smaller stripeCap paired with a larger block, but
//     never the reverse. Slicing block to stripeCap is therefore always in
//     bounds and yields stable per-index pointers.
//
// The zero value is a valid, empty stripe (stripeCap 0, nil block); the first
// get for any pID takes the slow path and grows it.
type stripe struct {
	mu Mutex // serializes stripe growth

	stripeCap atomic.Uint32           // current capacity (number of counters)
	block     atomic.Pointer[*uint64] // stripe block
}

// newStripe returns a new stripe initialized with a capacity
// appropriate for GOMAXPROCS length of counters.
func newStripe() *stripe {
	pCount := runtime.GOMAXPROCS(0)
	shards := make(stripeArr, pCount)
	var cnt int
	for intPtr := range allocCounters(pCount) {
		shards[cnt] = intPtr
		cnt++
	}
	s := new(stripe)
	s.block.Store(&shards[0])
	s.stripeCap.Store(uint32(pCount))
	return s
}

// get returns the counter associated with the current P.
// The caller must have the current P pinned (runtime_procPin).
// If the request pID is larger than the capacity for the stripe,
// get grows the stripe to accommodate it. We assume that the number
// of Ps has grown. This may temporarily unpin the P, pID may not
// be valid on return.
func (s *stripe) get(pID int) *uint64 {
	stripeCap := s.cap()

	if pID < stripeCap {
		// Fast path: lock-free read
		return unsafe.Slice(s.block.Load(), stripeCap)[pID]
	}
	return s.getSlow(pID, stripeCap)
}

// getSlow is the slow path to returning a counter associated
// with the current P. This is called when the stripe capacity
// needs to grow. The caller must have the current P pinned
// (runtime_procPin). This may temporarily unpin the P, pID may not
// be valid on return.
func (s *stripe) getSlow(pID, stripeCap int) *uint64 {
	for {
		if pID < stripeCap {
			// Fast path: lock-free read
			return unsafe.Slice(s.block.Load(), stripeCap)[pID]
		}

		// Slow path: grow
		// Unpin from the P to grab the lock.
		runtime_procUnpin()
		s.mu.Lock()

		// Check again under lock
		stripeCap = s.cap()
		if pID < stripeCap {
			// Another goroutine must have grown the stripe before
			// this goroutine obtained the lock. Release the lock
			// and attempt to return the counter.
			s.mu.Unlock()
			// Pin to a P again and update pID before retrying
			// the counter retrieval. The pID may have changed.
			pID = runtime_procPin()
			// Retry the counter retrieval.
			continue
		}
		// Grow the stripe, then refresh the local capacity so the retry
		// below can usually take the fast path without re-locking.
		s.grow()
		stripeCap = s.cap()
		s.mu.Unlock()

		// pin to a P again and update pID before retrying
		// the counter retrieval.
		pID = runtime_procPin()
	}
}

// grow grows the stripe to GOMAXPROCS if needed.
// s.mu must be held. Must not be pinned to a P while growing.
func (s *stripe) grow() {
	origStripeCap := s.cap()
	targetStripeLen := runtime.GOMAXPROCS(0)
	if targetStripeLen <= origStripeCap {
		return
	}
	block := unsafe.Slice(s.block.Load(), origStripeCap)
	newStripe := make(stripeArr, targetStripeLen)
	// Copy existing shards over.
	for idx := range len(block) {
		newStripe[idx] = block[idx]
	}
	var cnt int
	// allocCounters allocates one slot per P. When growing, keep the existing
	// pointers for Ps < origCnt so in-flight writes continue uninterrupted, and
	// discard the duplicate new pointers. Wasting a few slots on growth avoids
	// the complexity of tracking per-P allocation offsets.
	for intPtr := range allocCounters(targetStripeLen) {
		if cnt < origStripeCap {
			// skip over previously allocated ptr. Do
			// not set pointers allocated to the wrong pID.
			cnt++
			continue
		}
		newStripe[cnt] = intPtr
		cnt++
	}
	// block is always stored before stripeCap, and stripeCap is always
	// loaded before block. Assuming that block can only grow
	// then even if the reader sees an inconsistent block/stripeCap
	// combination, the access is safe since the logical size of
	// block will always be greater than or equal to the observed size
	// of stripeCap.
	s.block.Store(&newStripe[0])
	s.stripeCap.Store(uint32(targetStripeLen))
}

// cap returns the capacity of the stripe.
func (s *stripe) cap() int {
	return int(s.stripeCap.Load())
}
