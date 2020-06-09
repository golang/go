// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

const pageCachePages = 8 * unsafe.Sizeof(pageCache{}.cache)

// pageCache represents a per-p cache of pages the allocator can
// allocate from without a lock. More specifically, it represents
// a pageCachePages*pageSize chunk of memory with 0 or more free
// pages in it.
type pageCache struct {
	base  uintptr // base address of the chunk
	cache uint64  // 64-bit bitmap representing free pages (1 means free)
	scav  uint64  // 64-bit bitmap representing scavenged pages (1 means scavenged)
}

// empty returns true if the pageCache has any free pages, and false
// otherwise.
func (c *pageCache) empty() bool {
	return c.cache == 0
}

// alloc allocates npages from the page cache and is the main entry
// point for allocation.
//
// Returns a base address and the amount of scavenged memory in the
// allocated region in bytes.
//
// Returns a base address of zero on failure, in which case the
// amount of scavenged memory should be ignored.
func (c *pageCache) alloc(npages uintptr) (uintptr, uintptr) {
	if c.cache == 0 {
		return 0, 0
	}
	if npages == 1 {
		i := uintptr(sys.TrailingZeros64(c.cache))
		scav := (c.scav >> i) & 1
		c.cache &^= 1 << i // set bit to mark in-use
		c.scav &^= 1 << i  // clear bit to mark unscavenged
		return c.base + i*pageSize, uintptr(scav) * pageSize
	}
	return c.allocN(npages)
}

// allocN is a helper which attempts to allocate npages worth of pages
// from the cache. It represents the general case for allocating from
// the page cache.
//
// Returns a base address and the amount of scavenged memory in the
// allocated region in bytes.
func (c *pageCache) allocN(npages uintptr) (uintptr, uintptr) {
	i := findBitRange64(c.cache, uint(npages))
	if i >= 64 {
		return 0, 0
	}
	mask := ((uint64(1) << npages) - 1) << i
	scav := sys.OnesCount64(c.scav & mask)
	c.cache &^= mask // mark in-use bits
	c.scav &^= mask  // clear scavenged bits
	return c.base + uintptr(i*pageSize), uintptr(scav) * pageSize
}

// flush empties out unallocated free pages in the given cache
// into s. Then, it clears the cache, such that empty returns
// true.
//
// s.mheapLock must be held or the world must be stopped.
func (c *pageCache) flush(s *pageAlloc) {
	if c.empty() {
		return
	}
	ci := chunkIndex(c.base)
	pi := chunkPageIndex(c.base)

	// This method is called very infrequently, so just do the
	// slower, safer thing by iterating over each bit individually.
	for i := uint(0); i < 64; i++ {
		if c.cache&(1<<i) != 0 {
			s.chunkOf(ci).free1(pi + i)
		}
		if c.scav&(1<<i) != 0 {
			s.chunkOf(ci).scavenged.setRange(pi+i, 1)
		}
	}
	// Since this is a lot like a free, we need to make sure
	// we update the searchAddr just like free does.
	if s.compareSearchAddrTo(c.base) < 0 {
		s.searchAddr = c.base
	}
	s.update(c.base, pageCachePages, false, false)
	*c = pageCache{}
}

// allocToCache acquires a pageCachePages-aligned chunk of free pages which
// may not be contiguous, and returns a pageCache structure which owns the
// chunk.
//
// s.mheapLock must be held.
func (s *pageAlloc) allocToCache() pageCache {
	// If the searchAddr refers to a region which has a higher address than
	// any known chunk, then we know we're out of memory.
	if chunkIndex(s.searchAddr) >= s.end {
		return pageCache{}
	}
	c := pageCache{}
	ci := chunkIndex(s.searchAddr) // chunk index
	if s.summary[len(s.summary)-1][ci] != 0 {
		// Fast path: there's free pages at or near the searchAddr address.
		chunk := s.chunkOf(ci)
		j, _ := chunk.find(1, chunkPageIndex(s.searchAddr))
		if j < 0 {
			throw("bad summary data")
		}
		c = pageCache{
			base:  chunkBase(ci) + alignDown(uintptr(j), 64)*pageSize,
			cache: ^chunk.pages64(j),
			scav:  chunk.scavenged.block64(j),
		}
	} else {
		// Slow path: the searchAddr address had nothing there, so go find
		// the first free page the slow way.
		addr, _ := s.find(1)
		if addr == 0 {
			// We failed to find adequate free space, so mark the searchAddr as OoM
			// and return an empty pageCache.
			s.searchAddr = maxSearchAddr
			return pageCache{}
		}
		ci := chunkIndex(addr)
		chunk := s.chunkOf(ci)
		c = pageCache{
			base:  alignDown(addr, 64*pageSize),
			cache: ^chunk.pages64(chunkPageIndex(addr)),
			scav:  chunk.scavenged.block64(chunkPageIndex(addr)),
		}
	}

	// Set the bits as allocated and clear the scavenged bits.
	s.allocRange(c.base, pageCachePages)

	// Update as an allocation, but note that it's not contiguous.
	s.update(c.base, pageCachePages, false, true)

	// Set the search address to the last page represented by the cache.
	// Since all of the pages in this block are going to the cache, and we
	// searched for the first free page, we can confidently start at the
	// next page.
	//
	// However, s.searchAddr is not allowed to point into unmapped heap memory
	// unless it is maxSearchAddr, so make it the last page as opposed to
	// the page after.
	s.searchAddr = c.base + pageSize*(pageCachePages-1)
	return c
}
