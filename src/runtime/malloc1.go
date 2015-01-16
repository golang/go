// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// See malloc.h for overview.
//
// TODO(rsc): double-check stats.

package runtime

import "unsafe"

const _MaxArena32 = 2 << 30

// For use by Go. If it were a C enum it would be made available automatically,
// but the value of MaxMem is too large for enum.
// XXX - uintptr runtime·maxmem = MaxMem;

func mlookup(v uintptr, base *uintptr, size *uintptr, sp **mspan) int32 {
	_g_ := getg()

	_g_.m.mcache.local_nlookup++
	if ptrSize == 4 && _g_.m.mcache.local_nlookup >= 1<<30 {
		// purge cache stats to prevent overflow
		lock(&mheap_.lock)
		purgecachedstats(_g_.m.mcache)
		unlock(&mheap_.lock)
	}

	s := mHeap_LookupMaybe(&mheap_, unsafe.Pointer(v))
	if sp != nil {
		*sp = s
	}
	if s == nil {
		if base != nil {
			*base = 0
		}
		if size != nil {
			*size = 0
		}
		return 0
	}

	p := uintptr(s.start) << _PageShift
	if s.sizeclass == 0 {
		// Large object.
		if base != nil {
			*base = p
		}
		if size != nil {
			*size = s.npages << _PageShift
		}
		return 1
	}

	n := s.elemsize
	if base != nil {
		i := (uintptr(v) - uintptr(p)) / n
		*base = p + i*n
	}
	if size != nil {
		*size = n
	}

	return 1
}

//go:nosplit
func purgecachedstats(c *mcache) {
	// Protected by either heap or GC lock.
	h := &mheap_
	memstats.heap_alloc += uint64(c.local_cachealloc)
	c.local_cachealloc = 0
	memstats.tinyallocs += uint64(c.local_tinyallocs)
	c.local_tinyallocs = 0
	memstats.nlookup += uint64(c.local_nlookup)
	c.local_nlookup = 0
	h.largefree += uint64(c.local_largefree)
	c.local_largefree = 0
	h.nlargefree += uint64(c.local_nlargefree)
	c.local_nlargefree = 0
	for i := 0; i < len(c.local_nsmallfree); i++ {
		h.nsmallfree[i] += uint64(c.local_nsmallfree[i])
		c.local_nsmallfree[i] = 0
	}
}

func mallocinit() {
	initSizes()

	if class_to_size[_TinySizeClass] != _TinySize {
		throw("bad TinySizeClass")
	}

	var p, bitmapSize, spansSize, pSize, limit uintptr
	var reserved bool

	// limit = runtime.memlimit();
	// See https://code.google.com/p/go/issues/detail?id=5049
	// TODO(rsc): Fix after 1.1.
	limit = 0

	// Set up the allocation arena, a contiguous area of memory where
	// allocated data will be found.  The arena begins with a bitmap large
	// enough to hold 4 bits per allocated word.
	if ptrSize == 8 && (limit == 0 || limit > 1<<30) {
		// On a 64-bit machine, allocate from a single contiguous reservation.
		// 128 GB (MaxMem) should be big enough for now.
		//
		// The code will work with the reservation at any address, but ask
		// SysReserve to use 0x0000XXc000000000 if possible (XX=00...7f).
		// Allocating a 128 GB region takes away 37 bits, and the amd64
		// doesn't let us choose the top 17 bits, so that leaves the 11 bits
		// in the middle of 0x00c0 for us to choose.  Choosing 0x00c0 means
		// that the valid memory addresses will begin 0x00c0, 0x00c1, ..., 0x00df.
		// In little-endian, that's c0 00, c1 00, ..., df 00. None of those are valid
		// UTF-8 sequences, and they are otherwise as far away from
		// ff (likely a common byte) as possible.  If that fails, we try other 0xXXc0
		// addresses.  An earlier attempt to use 0x11f8 caused out of memory errors
		// on OS X during thread allocations.  0x00c0 causes conflicts with
		// AddressSanitizer which reserves all memory up to 0x0100.
		// These choices are both for debuggability and to reduce the
		// odds of the conservative garbage collector not collecting memory
		// because some non-pointer block of memory had a bit pattern
		// that matched a memory address.
		//
		// Actually we reserve 136 GB (because the bitmap ends up being 8 GB)
		// but it hardly matters: e0 00 is not valid UTF-8 either.
		//
		// If this fails we fall back to the 32 bit memory mechanism
		arenaSize := round(_MaxMem, _PageSize)
		bitmapSize = arenaSize / (ptrSize * 8 / 4)
		spansSize = arenaSize / _PageSize * ptrSize
		spansSize = round(spansSize, _PageSize)
		for i := 0; i <= 0x7f; i++ {
			p = uintptr(i)<<40 | uintptrMask&(0x00c0<<32)
			pSize = bitmapSize + spansSize + arenaSize + _PageSize
			p = uintptr(sysReserve(unsafe.Pointer(p), pSize, &reserved))
			if p != 0 {
				break
			}
		}
	}

	if p == 0 {
		// On a 32-bit machine, we can't typically get away
		// with a giant virtual address space reservation.
		// Instead we map the memory information bitmap
		// immediately after the data segment, large enough
		// to handle another 2GB of mappings (256 MB),
		// along with a reservation for an initial arena.
		// When that gets used up, we'll start asking the kernel
		// for any memory anywhere and hope it's in the 2GB
		// following the bitmap (presumably the executable begins
		// near the bottom of memory, so we'll have to use up
		// most of memory before the kernel resorts to giving out
		// memory before the beginning of the text segment).
		//
		// Alternatively we could reserve 512 MB bitmap, enough
		// for 4GB of mappings, and then accept any memory the
		// kernel threw at us, but normally that's a waste of 512 MB
		// of address space, which is probably too much in a 32-bit world.

		// If we fail to allocate, try again with a smaller arena.
		// This is necessary on Android L where we share a process
		// with ART, which reserves virtual memory aggressively.
		arenaSizes := []uintptr{
			512 << 20,
			256 << 20,
		}

		for _, arenaSize := range arenaSizes {
			bitmapSize = _MaxArena32 / (ptrSize * 8 / 4)
			spansSize = _MaxArena32 / _PageSize * ptrSize
			if limit > 0 && arenaSize+bitmapSize+spansSize > limit {
				bitmapSize = (limit / 9) &^ ((1 << _PageShift) - 1)
				arenaSize = bitmapSize * 8
				spansSize = arenaSize / _PageSize * ptrSize
			}
			spansSize = round(spansSize, _PageSize)

			// SysReserve treats the address we ask for, end, as a hint,
			// not as an absolute requirement.  If we ask for the end
			// of the data segment but the operating system requires
			// a little more space before we can start allocating, it will
			// give out a slightly higher pointer.  Except QEMU, which
			// is buggy, as usual: it won't adjust the pointer upward.
			// So adjust it upward a little bit ourselves: 1/4 MB to get
			// away from the running binary image and then round up
			// to a MB boundary.
			p = round(uintptr(unsafe.Pointer(&end))+(1<<18), 1<<20)
			pSize = bitmapSize + spansSize + arenaSize + _PageSize
			p = uintptr(sysReserve(unsafe.Pointer(p), pSize, &reserved))
			if p != 0 {
				break
			}
		}
		if p == 0 {
			throw("runtime: cannot reserve arena virtual address space")
		}
	}

	// PageSize can be larger than OS definition of page size,
	// so SysReserve can give us a PageSize-unaligned pointer.
	// To overcome this we ask for PageSize more and round up the pointer.
	p1 := round(p, _PageSize)

	mheap_.spans = (**mspan)(unsafe.Pointer(p1))
	mheap_.bitmap = p1 + spansSize
	mheap_.arena_start = p1 + (spansSize + bitmapSize)
	mheap_.arena_used = mheap_.arena_start
	mheap_.arena_end = p + pSize
	mheap_.arena_reserved = reserved

	if mheap_.arena_start&(_PageSize-1) != 0 {
		println("bad pagesize", hex(p), hex(p1), hex(spansSize), hex(bitmapSize), hex(_PageSize), "start", hex(mheap_.arena_start))
		throw("misrounded allocation in mallocinit")
	}

	// Initialize the rest of the allocator.
	mHeap_Init(&mheap_, spansSize)
	_g_ := getg()
	_g_.m.mcache = allocmcache()
}

// sysReserveHigh reserves space somewhere high in the address space.
// sysReserve doesn't actually reserve the full amount requested on
// 64-bit systems, because of problems with ulimit. Instead it checks
// that it can get the first 64 kB and assumes it can grab the rest as
// needed. This doesn't work well with the "let the kernel pick an address"
// mode, so don't do that. Pick a high address instead.
func sysReserveHigh(n uintptr, reserved *bool) unsafe.Pointer {
	if ptrSize == 4 {
		return sysReserve(nil, n, reserved)
	}

	for i := 0; i <= 0x7f; i++ {
		p := uintptr(i)<<40 | uintptrMask&(0x00c0<<32)
		*reserved = false
		p = uintptr(sysReserve(unsafe.Pointer(p), n, reserved))
		if p != 0 {
			return unsafe.Pointer(p)
		}
	}

	return sysReserve(nil, n, reserved)
}

func mHeap_SysAlloc(h *mheap, n uintptr) unsafe.Pointer {
	if n > uintptr(h.arena_end)-uintptr(h.arena_used) {
		// We are in 32-bit mode, maybe we didn't use all possible address space yet.
		// Reserve some more space.
		p_size := round(n+_PageSize, 256<<20)
		new_end := h.arena_end + p_size
		if new_end <= h.arena_start+_MaxArena32 {
			// TODO: It would be bad if part of the arena
			// is reserved and part is not.
			var reserved bool
			p := uintptr(sysReserve((unsafe.Pointer)(h.arena_end), p_size, &reserved))
			if p == h.arena_end {
				h.arena_end = new_end
				h.arena_reserved = reserved
			} else if p+p_size <= h.arena_start+_MaxArena32 {
				// Keep everything page-aligned.
				// Our pages are bigger than hardware pages.
				h.arena_end = p + p_size
				h.arena_used = p + (-uintptr(p) & (_PageSize - 1))
				h.arena_reserved = reserved
			} else {
				var stat uint64
				sysFree((unsafe.Pointer)(p), p_size, &stat)
			}
		}
	}

	if n <= uintptr(h.arena_end)-uintptr(h.arena_used) {
		// Keep taking from our reservation.
		p := h.arena_used
		sysMap((unsafe.Pointer)(p), n, h.arena_reserved, &memstats.heap_sys)
		h.arena_used += n
		mHeap_MapBits(h)
		mHeap_MapSpans(h)
		if raceenabled {
			racemapshadow((unsafe.Pointer)(p), n)
		}
		if mheap_.shadow_enabled {
			sysMap(unsafe.Pointer(p+mheap_.shadow_heap), n, h.shadow_reserved, &memstats.other_sys)
		}

		if uintptr(p)&(_PageSize-1) != 0 {
			throw("misrounded allocation in MHeap_SysAlloc")
		}
		return (unsafe.Pointer)(p)
	}

	// If using 64-bit, our reservation is all we have.
	if uintptr(h.arena_end)-uintptr(h.arena_start) >= _MaxArena32 {
		return nil
	}

	// On 32-bit, once the reservation is gone we can
	// try to get memory at a location chosen by the OS
	// and hope that it is in the range we allocated bitmap for.
	p_size := round(n, _PageSize) + _PageSize
	p := uintptr(sysAlloc(p_size, &memstats.heap_sys))
	if p == 0 {
		return nil
	}

	if p < h.arena_start || uintptr(p)+p_size-uintptr(h.arena_start) >= _MaxArena32 {
		print("runtime: memory allocated by OS (", p, ") not in usable range [", hex(h.arena_start), ",", hex(h.arena_start+_MaxArena32), ")\n")
		sysFree((unsafe.Pointer)(p), p_size, &memstats.heap_sys)
		return nil
	}

	p_end := p + p_size
	p += -p & (_PageSize - 1)
	if uintptr(p)+n > uintptr(h.arena_used) {
		h.arena_used = p + n
		if p_end > h.arena_end {
			h.arena_end = p_end
		}
		mHeap_MapBits(h)
		mHeap_MapSpans(h)
		if raceenabled {
			racemapshadow((unsafe.Pointer)(p), n)
		}
	}

	if uintptr(p)&(_PageSize-1) != 0 {
		throw("misrounded allocation in MHeap_SysAlloc")
	}
	return (unsafe.Pointer)(p)
}

var end struct{}

func largeAlloc(size uintptr, flag uint32) *mspan {
	// print("largeAlloc size=", size, "\n")

	if size+_PageSize < size {
		throw("out of memory")
	}
	npages := size >> _PageShift
	if size&_PageMask != 0 {
		npages++
	}
	s := mHeap_Alloc(&mheap_, npages, 0, true, flag&_FlagNoZero == 0)
	if s == nil {
		throw("out of memory")
	}
	s.limit = uintptr(s.start)<<_PageShift + size
	heapBitsForSpan(s.base()).initSpan(s.layout())
	return s
}
