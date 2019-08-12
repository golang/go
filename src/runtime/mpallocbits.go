// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"math/bits"
)

// pageBits is a bitmap representing one bit per page in a palloc chunk.
type pageBits [pallocChunkPages / 64]uint64

// get returns the value of the i'th bit in the bitmap.
func (b *pageBits) get(i uint) uint {
	return uint((b[i/64] >> (i % 64)) & 1)
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
	for k := i/64 + 1; k < j/64; k++ {
		b[k] = 0
	}
	// Clear trailing bits.
	b[j/64] &^= (uint64(1) << (j%64 + 1)) - 1
}

// clearAll frees all the bits of b.
func (b *pageBits) clearAll() {
	for i := range b {
		b[i] = 0
	}
}

// pallocBits is a bitmap that tracks page allocations for at most one
// palloc chunk.
//
// The precise representation is an implementation detail, but for the
// sake of documentation, 0s are free pages and 1s are allocated pages.
type pallocBits pageBits

// find searches for npages contiguous free pages in pallocBits and returns
// the index where that run starts, as well as the index of the first free page
// it found in the search. searchIdx represents the first known free page and
// where to begin the search from.
//
// If find fails to find any free space, it returns an index of ^uint(0) and
// the new searchIdx should be ignored.
//
// The returned searchIdx is always the index of the first free page found
// in this bitmap during the search, except if npages == 1, in which
// case it will be the index just after the first free page, because the
// index returned as the first result is assumed to be allocated and so
// represents a minor optimization for that case.
func (b *pallocBits) find(npages uintptr, searchIdx uint) (uint, uint) {
	if npages == 1 {
		addr := b.find1(searchIdx)
		// Return a searchIdx of addr + 1 since we assume addr will be
		// allocated.
		return addr, addr + 1
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
	for i := searchIdx / 64; i < uint(len(b)); i++ {
		x := b[i]
		if x == ^uint64(0) {
			continue
		}
		return i*64 + uint(bits.TrailingZeros64(^x))
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
		if bi == ^uint64(0) {
			end = 0
			continue
		}
		// First see if we can pack our allocation in the trailing
		// zeros plus the end of the last 64 bits.
		start := uint(bits.TrailingZeros64(bi))
		if newSearchIdx == ^uint(0) {
			// The new searchIdx is going to be at these 64 bits after any
			// 1s we file, so count trailing 1s.
			newSearchIdx = i*64 + uint(bits.TrailingZeros64(^bi))
		}
		if end+start >= uint(npages) {
			return i*64 - end, newSearchIdx
		}
		// Next, check the interior of the 64-bit chunk.
		j := findBitRange64(^bi, uint(npages))
		if j < 64 {
			return i*64 + j, newSearchIdx
		}
		end = uint(bits.LeadingZeros64(bi))
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
			newSearchIdx = i*64 + uint(bits.TrailingZeros64(^x))
		}
		if size == 0 {
			size = uint(bits.LeadingZeros64(x))
			start = i*64 + 64 - size
			continue
		}
		s := uint(bits.TrailingZeros64(x))
		if s+size >= uint(npages) {
			size += s
			return start, newSearchIdx
		}
		if s < 64 {
			size = uint(bits.LeadingZeros64(x))
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

// findBitRange64 returns the bit index of the first set of
// n consecutive 1 bits. If no consecutive set of 1 bits of
// size n may be found in c, then it returns an integer >= 64.
func findBitRange64(c uint64, n uint) uint {
	i := uint(0)
	cont := uint(bits.TrailingZeros64(^c))
	for cont < n && i < 64 {
		i += cont
		i += uint(bits.TrailingZeros64(c >> i))
		cont = uint(bits.TrailingZeros64(^(c >> i)))
	}
	return i
}
