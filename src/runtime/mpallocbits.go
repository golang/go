// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
)

// pageBits is a bitmap representing one bit per page in a palloc chunk.
type pageBits [pallocChunkPages / 64]uint64

// get returns the value of the i'th bit in the bitmap.
func (b *pageBits) get(i uint) uint {
	return uint((b[i/64] >> (i % 64)) & 1)
}

// block64 returns the 64-bit aligned block of bits containing the i'th bit.
func (b *pageBits) block64(i uint) uint64 {
	return b[i/64]
}

// set sets bit i of pageBits.
func (b *pageBits) set(i uint) {
	b[i/64] |= 1 << (i % 64)
}

// setRange sets bits in the range [i, i+n).
func (b *pageBits) setRange(i, n uint) {
	_ = b[i/64]
	if n == 1 {
		// Fast path for the n == 1 case.
		b.set(i)
		return
	}
	// Set bits [i, j].
	j := i + n - 1
	if i/64 == j/64 {
		b[i/64] |= ((uint64(1) << n) - 1) << (i % 64)
		return
	}
	_ = b[j/64]
	// Set leading bits.
	b[i/64] |= ^uint64(0) << (i % 64)
	for k := i/64 + 1; k < j/64; k++ {
		b[k] = ^uint64(0)
	}
	// Set trailing bits.
	b[j/64] |= (uint64(1) << (j%64 + 1)) - 1
}

// setAll sets all the bits of b.
func (b *pageBits) setAll() {
	for i := range b {
		b[i] = ^uint64(0)
	}
}

// setBlock64 sets the 64-bit aligned block of bits containing the i'th bit that
// are set in v.
func (b *pageBits) setBlock64(i uint, v uint64) {
	b[i/64] |= v
}

// clear clears bit i of pageBits.
func (b *pageBits) clear(i uint) {
	b[i/64] &^= 1 << (i % 64)
}

// clearRange clears bits in the range [i, i+n).
func (b *pageBits) clearRange(i, n uint) {
	_ = b[i/64]
	if n == 1 {
		// Fast path for the n == 1 case.
		b.clear(i)
		return
	}
	// Clear bits [i, j].
	j := i + n - 1
	if i/64 == j/64 {
		b[i/64] &^= ((uint64(1) << n) - 1) << (i % 64)
		return
	}
	_ = b[j/64]
	// Clear leading bits.
	b[i/64] &^= ^uint64(0) << (i % 64)
	clear(b[i/64+1 : j/64])
	// Clear trailing bits.
	b[j/64] &^= (uint64(1) << (j%64 + 1)) - 1
}

// clearAll frees all the bits of b.
func (b *pageBits) clearAll() {
	clear(b[:])
}

// clearBlock64 clears the 64-bit aligned block of bits containing the i'th bit that
// are set in v.
func (b *pageBits) clearBlock64(i uint, v uint64) {
	b[i/64] &^= v
}

// popcntRange counts the number of set bits in the
// range [i, i+n).
func (b *pageBits) popcntRange(i, n uint) (s uint) {
	if n == 1 {
		return uint((b[i/64] >> (i % 64)) & 1)
	}
	_ = b[i/64]
	j := i + n - 1
	if i/64 == j/64 {
		return uint(sys.OnesCount64((b[i/64] >> (i % 64)) & ((1 << n) - 1)))
	}
	_ = b[j/64]
	s += uint(sys.OnesCount64(b[i/64] >> (i % 64)))
	for k := i/64 + 1; k < j/64; k++ {
		s += uint(sys.OnesCount64(b[k]))
	}
	s += uint(sys.OnesCount64(b[j/64] & ((1 << (j%64 + 1)) - 1)))
	return
}

// pallocBits is a bitmap that tracks page allocations for at most one
// palloc chunk.
//
// The precise representation is an implementation detail, but for the
// sake of documentation, 0s are free pages and 1s are allocated pages.
type pallocBits pageBits

// summarize returns a packed summary of the bitmap in pallocBits.
func (b *pallocBits) summarize() pallocSum {
	var start, most, cur uint
	const notSetYet = ^uint(0) // sentinel for start value
	start = notSetYet
	for i := 0; i < len(b); i++ {
		x := b[i]
		if x == 0 {
			cur += 64
			continue
		}
		t := uint(sys.TrailingZeros64(x))
		l := uint(sys.LeadingZeros64(x))

		// Finish any region spanning the uint64s
		cur += t
		if start == notSetYet {
			start = cur
		}
		most = max(most, cur)
		// Final region that might span to next uint64
		cur = l
	}
	if start == notSetYet {
		// Made it all the way through without finding a single 1 bit.
		const n = uint(64 * len(b))
		return packPallocSum(n, n, n)
	}
	most = max(most, cur)

	if most >= 64-2 {
		// There is no way an internal run of zeros could beat max.
		return packPallocSum(start, most, cur)
	}
	// Now look inside each uint64 for runs of zeros.
	// All uint64s must be nonzero, or we would have aborted above.
outer:
	for i := 0; i < len(b); i++ {
		x := b[i]

		// Look inside this uint64. We have a pattern like
		// 000000 1xxxxx1 000000
		// We need to look inside the 1xxxxx1 for any contiguous
		// region of zeros.

		// We already know the trailing zeros are no larger than max. Remove them.
		x >>= sys.TrailingZeros64(x) & 63
		if x&(x+1) == 0 { // no more zeros (except at the top).
			continue
		}

		// Strategy: shrink all runs of zeros by max. If any runs of zero
		// remain, then we've identified a larger maximum zero run.
		p := most    // number of zeros we still need to shrink by.
		k := uint(1) // current minimum length of runs of ones in x.
		for {
			// Shrink all runs of zeros by p places (except the top zeros).
			for p > 0 {
				if p <= k {
					// Shift p ones down into the top of each run of zeros.
					x |= x >> (p & 63)
					if x&(x+1) == 0 { // no more zeros (except at the top).
						continue outer
					}
					break
				}
				// Shift k ones down into the top of each run of zeros.
				x |= x >> (k & 63)
				if x&(x+1) == 0 { // no more zeros (except at the top).
					continue outer
				}
				p -= k
				// We've just doubled the minimum length of 1-runs.
				// This allows us to shift farther in the next iteration.
				k *= 2
			}

			// The length of the lowest-order zero run is an increment to our maximum.
			j := uint(sys.TrailingZeros64(^x)) // count contiguous trailing ones
			x >>= j & 63                       // remove trailing ones
			j = uint(sys.TrailingZeros64(x))   // count contiguous trailing zeros
			x >>= j & 63                       // remove zeros
			most += j                          // we have a new maximum!
			if x&(x+1) == 0 {                  // no more zeros (except at the top).
				continue outer
			}
			p = j // remove j more zeros from each zero run.
		}
	}
	return packPallocSum(start, most, cur)
}

// find searches for npages contiguous free pages in pallocBits and returns
// the index where that run starts, as well as the index of the first free page
// it found in the search. searchIdx represents the first known free page and
// where to begin the next search from.
//
// If find fails to find any free space, it returns an index of ^uint(0) and
// the new searchIdx should be ignored.
//
// Note that if npages == 1, the two returned values will always be identical.
func (b *pallocBits) find(npages uintptr, searchIdx uint) (uint, uint) {
	if npages == 1 {
		addr := b.find1(searchIdx)
		return addr, addr
	} else if npages <= 64 {
		return b.findSmallN(npages, searchIdx)
	}
	return b.findLargeN(npages, searchIdx)
}

// find1 is a helper for find which searches for a single free page
// in the pallocBits and returns the index.
//
// See find for an explanation of the searchIdx parameter.
func (b *pallocBits) find1(searchIdx uint) uint {
	_ = b[0] // lift nil check out of loop
	for i := searchIdx / 64; i < uint(len(b)); i++ {
		x := b[i]
		if ^x == 0 {
			continue
		}
		return i*64 + uint(sys.TrailingZeros64(^x))
	}
	return ^uint(0)
}

// findSmallN is a helper for find which searches for npages contiguous free pages
// in this pallocBits and returns the index where that run of contiguous pages
// starts as well as the index of the first free page it finds in its search.
//
// See find for an explanation of the searchIdx parameter.
//
// Returns a ^uint(0) index on failure and the new searchIdx should be ignored.
//
// findSmallN assumes npages <= 64, where any such contiguous run of pages
// crosses at most one aligned 64-bit boundary in the bits.
func (b *pallocBits) findSmallN(npages uintptr, searchIdx uint) (uint, uint) {
	end, newSearchIdx := uint(0), ^uint(0)
	for i := searchIdx / 64; i < uint(len(b)); i++ {
		bi := b[i]
		if ^bi == 0 {
			end = 0
			continue
		}
		// First see if we can pack our allocation in the trailing
		// zeros plus the end of the last 64 bits.
		if newSearchIdx == ^uint(0) {
			// The new searchIdx is going to be at these 64 bits after any
			// 1s we file, so count trailing 1s.
			newSearchIdx = i*64 + uint(sys.TrailingZeros64(^bi))
		}
		start := uint(sys.TrailingZeros64(bi))
		if end+start >= uint(npages) {
			return i*64 - end, newSearchIdx
		}
		// Next, check the interior of the 64-bit chunk.
		j := findBitRange64(^bi, uint(npages))
		if j < 64 {
			return i*64 + j, newSearchIdx
		}
		end = uint(sys.LeadingZeros64(bi))
	}
	return ^uint(0), newSearchIdx
}

// findLargeN is a helper for find which searches for npages contiguous free pages
// in this pallocBits and returns the index where that run starts, as well as the
// index of the first free page it found it its search.
//
// See alloc for an explanation of the searchIdx parameter.
//
// Returns a ^uint(0) index on failure and the new searchIdx should be ignored.
//
// findLargeN assumes npages > 64, where any such run of free pages
// crosses at least one aligned 64-bit boundary in the bits.
func (b *pallocBits) findLargeN(npages uintptr, searchIdx uint) (uint, uint) {
	start, size, newSearchIdx := ^uint(0), uint(0), ^uint(0)
	for i := searchIdx / 64; i < uint(len(b)); i++ {
		x := b[i]
		if x == ^uint64(0) {
			size = 0
			continue
		}
		if newSearchIdx == ^uint(0) {
			// The new searchIdx is going to be at these 64 bits after any
			// 1s we file, so count trailing 1s.
			newSearchIdx = i*64 + uint(sys.TrailingZeros64(^x))
		}
		if size == 0 {
			size = uint(sys.LeadingZeros64(x))
			start = i*64 + 64 - size
			continue
		}
		s := uint(sys.TrailingZeros64(x))
		if s+size >= uint(npages) {
			return start, newSearchIdx
		}
		if s < 64 {
			size = uint(sys.LeadingZeros64(x))
			start = i*64 + 64 - size
			continue
		}
		size += 64
	}
	if size < uint(npages) {
		return ^uint(0), newSearchIdx
	}
	return start, newSearchIdx
}

// allocRange allocates the range [i, i+n).
func (b *pallocBits) allocRange(i, n uint) {
	(*pageBits)(b).setRange(i, n)
}

// allocAll allocates all the bits of b.
func (b *pallocBits) allocAll() {
	(*pageBits)(b).setAll()
}

// free1 frees a single page in the pallocBits at i.
func (b *pallocBits) free1(i uint) {
	(*pageBits)(b).clear(i)
}

// free frees the range [i, i+n) of pages in the pallocBits.
func (b *pallocBits) free(i, n uint) {
	(*pageBits)(b).clearRange(i, n)
}

// freeAll frees all the bits of b.
func (b *pallocBits) freeAll() {
	(*pageBits)(b).clearAll()
}

// pages64 returns a 64-bit bitmap representing a block of 64 pages aligned
// to 64 pages. The returned block of pages is the one containing the i'th
// page in this pallocBits. Each bit represents whether the page is in-use.
func (b *pallocBits) pages64(i uint) uint64 {
	return (*pageBits)(b).block64(i)
}

// allocPages64 allocates a 64-bit block of 64 pages aligned to 64 pages according
// to the bits set in alloc. The block set is the one containing the i'th page.
func (b *pallocBits) allocPages64(i uint, alloc uint64) {
	(*pageBits)(b).setBlock64(i, alloc)
}

// findBitRange64 returns the bit index of the first set of
// n consecutive 1 bits. If no consecutive set of 1 bits of
// size n may be found in c, then it returns an integer >= 64.
// n must be > 0.
func findBitRange64(c uint64, n uint) uint {
	// This implementation is based on shrinking the length of
	// runs of contiguous 1 bits. We remove the top n-1 1 bits
	// from each run of 1s, then look for the first remaining 1 bit.
	p := n - 1   // number of 1s we want to remove.
	k := uint(1) // current minimum width of runs of 0 in c.
	for p > 0 {
		if p <= k {
			// Shift p 0s down into the top of each run of 1s.
			c &= c >> (p & 63)
			break
		}
		// Shift k 0s down into the top of each run of 1s.
		c &= c >> (k & 63)
		if c == 0 {
			return 64
		}
		p -= k
		// We've just doubled the minimum length of 0-runs.
		// This allows us to shift farther in the next iteration.
		k *= 2
	}
	// Find first remaining 1.
	// Since we shrunk from the top down, the first 1 is in
	// its correct original position.
	return uint(sys.TrailingZeros64(c))
}

// pallocData encapsulates pallocBits and a bitmap for
// whether or not a given page is scavenged in a single
// structure. It's effectively a pallocBits with
// additional functionality.
//
// Update the comment on (*pageAlloc).chunks should this
// structure change.
type pallocData struct {
	pallocBits
	scavenged pageBits
}

// allocRange sets bits [i, i+n) in the bitmap to 1 and
// updates the scavenged bits appropriately.
func (m *pallocData) allocRange(i, n uint) {
	// Clear the scavenged bits when we alloc the range.
	m.pallocBits.allocRange(i, n)
	m.scavenged.clearRange(i, n)
}

// allocAll sets every bit in the bitmap to 1 and updates
// the scavenged bits appropriately.
func (m *pallocData) allocAll() {
	// Clear the scavenged bits when we alloc the range.
	m.pallocBits.allocAll()
	m.scavenged.clearAll()
}
