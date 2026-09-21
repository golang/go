// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// OS memory management abstraction layer
//
// Regions of the address space managed by the runtime may be in one of four
// states at any given time:
// 1) None - Unreserved and unmapped, the default state of any region.
// 2) Reserved - Owned by the runtime, but accessing it would cause a fault.
//               Does not count against the process' memory footprint.
// 3) Prepared - Reserved, intended not to be backed by physical memory (though
//               an OS may implement this lazily). Can transition efficiently to
//               Ready. Accessing memory in such a region is undefined (may
//               fault, may give back unexpected zeroes, etc.).
// 4) Ready - may be accessed safely.
//
// This set of states is more than strictly necessary to support all the
// currently supported platforms. One could get by with just None, Reserved, and
// Ready. However, the Prepared state gives us flexibility for performance
// purposes. For example, on POSIX-y operating systems, Reserved is usually a
// private anonymous mmap'd region with PROT_NONE set, and to transition
// to Ready would require setting PROT_READ|PROT_WRITE. However the
// underspecification of Prepared lets us use just MADV_FREE to transition from
// Ready to Prepared. Thus with the Prepared state we can set the permission
// bits just once early on, we can efficiently tell the OS that it's free to
// take pages away from us when we don't strictly need them.
//
// This file defines a cross-OS interface for a common set of helpers
// that transition memory regions between these states. The helpers call into
// OS-specific implementations that handle errors, while the interface boundary
// implements cross-OS functionality, like updating runtime accounting.

// sysAlloc transitions an OS-chosen region of memory from None to Ready.
// More specifically, it obtains a large chunk of zeroed memory from the
// operating system, typically on the order of a hundred kilobytes
// or a megabyte. This memory is always immediately available for use.
//
// sysStat must be non-nil.
//
// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//
//go:nosplit
func sysAlloc(n uintptr, sysStat *sysMemStat, vmaName string) unsafe.Pointer {
	sysStat.add(int64(n))
	gcController.mappedReady.Add(int64(n))
	p := sysAllocOS(n, vmaName)

	// When using ASAN leak detection, we must tell ASAN about
	// cases where we store pointers in mmapped memory.
	if asanenabled {
		lsanregisterrootregion(p, n)
	}

	return p
}

// sysUnused transitions a memory region from Ready to Prepared. It notifies the
// operating system that the physical pages backing this memory region are no
// longer needed and can be reused for other purposes. The contents of a
// sysUnused memory region are considered forfeit and the region must not be
// accessed again until sysUsed is called.
func sysUnused(v unsafe.Pointer, n uintptr) {
	gcController.mappedReady.Add(-int64(n))
	sysUnusedOS(v, n)
}

// needZeroAfterSysUnused reports whether memory returned by sysUnused must be
// zeroed for use.
func needZeroAfterSysUnused() bool {
	return needZeroAfterSysUnusedOS()
}

// sysUsed transitions a memory region from Prepared to Ready. It notifies the
// operating system that the memory region is needed and ensures that the region
// may be safely accessed. This is typically a no-op on systems that don't have
// an explicit commit step and hard over-commit limits, but is critical on
// Windows, for example.
//
// This operation is idempotent for memory already in the Prepared state, so
// it is safe to refer, with v and n, to a range of memory that includes both
// Prepared and Ready memory. However, the caller must provide the exact amount
// of Prepared memory for accounting purposes.
func sysUsed(v unsafe.Pointer, n, prepared uintptr) {
	gcController.mappedReady.Add(int64(prepared))
	sysUsedOS(v, n)
}

// sysHugePage does not transition memory regions, but instead provides a
// hint to the OS that it would be more efficient to back this memory region
// with pages of a larger size transparently.
func sysHugePage(v unsafe.Pointer, n uintptr) {
	sysHugePageOS(v, n)
}

// sysNoHugePage does not transition memory regions, but instead provides a
// hint to the OS that it would be less efficient to back this memory region
// with pages of a larger size transparently.
func sysNoHugePage(v unsafe.Pointer, n uintptr) {
	sysNoHugePageOS(v, n)
}

// sysHugePageCollapse attempts to immediately back the provided memory region
// with huge pages. It is best-effort and may fail silently.
func sysHugePageCollapse(v unsafe.Pointer, n uintptr) {
	sysHugePageCollapseOS(v, n)
}

// sysFree transitions a memory region from Ready to None. Therefore, it
// returns memory unconditionally.
//
// sysStat must be non-nil.
//
// The size and start address must exactly match the size and returned address
// from the original sysAlloc/sysReserve/sysReserveAligned call. That is,
// sysFree cannot be used to free a subset of a memory region.
//
// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//
//go:nosplit
func sysFree(v unsafe.Pointer, n uintptr, sysStat *sysMemStat) {
	// When using ASAN leak detection, the memory being freed is known by
	// the sanitizer. We need to unregister it so it's not accessed by it.
	//
	// lsanunregisterrootregion matches regions by start address and size,
	// so it is not possible to unregister a subset of the region. This is
	// why sysFree requires the full region from the initial allocation.
	if asanenabled {
		lsanunregisterrootregion(v, n)
	}

	sysStat.add(-int64(n))
	gcController.mappedReady.Add(-int64(n))
	sysFreeOS(v, n)
}

// sysFault transitions a memory region from Ready to Reserved. It
// marks a region such that it will always fault if accessed. Used only for
// debugging the runtime.
//
// TODO(mknyszek): Currently it's true that all uses of sysFault transition
// memory from Ready to Reserved, but this may not be true in the future
// since on every platform the operation is much more general than that.
// If a transition from Prepared is ever introduced, create a new function
// that elides the Ready state accounting.
func sysFault(v unsafe.Pointer, n uintptr) {
	gcController.mappedReady.Add(-int64(n))
	sysFaultOS(v, n)
}

// sysReserve transitions a memory region from None to Reserved. It reserves
// address space in such a way that it would cause a fatal fault upon access
// (either via permissions or not committing the memory). Such a reservation is
// thus never backed by physical memory.
//
// If the pointer passed to it is non-nil, the caller wants the reservation
// there, but sysReserve can still choose another location if that one is
// unavailable.
//
// sysReserve returns OS-aligned memory. If a larger alignment is required, use
// sysReservedAligned.
func sysReserve(v unsafe.Pointer, n uintptr, vmaName string) unsafe.Pointer {
	p := sysReserveOS(v, n, vmaName)

	// When using ASAN leak detection, we must tell ASAN about
	// cases where we store pointers in mmapped memory.
	if asanenabled {
		lsanregisterrootregion(p, n)
	}

	return p
}

// sysReserveAligned transitions a memory region from None to Reserved.
//
// Semantics are equivlent to sysReserve, but the returned pointer is aligned
// to align bytes. It may reserve either n or n+align bytes, so it returns the
// size that was reserved.
func sysReserveAligned(v unsafe.Pointer, size, align uintptr, vmaName string) (unsafe.Pointer, uintptr) {
	if isSbrkPlatform {
		if v != nil {
			throw("unexpected heap arena hint on sbrk platform")
		}
		return sysReserveAlignedSbrk(size, align)
	}
	// Since the alignment is rather large in uses of this
	// function, we're not likely to get it by chance, so we ask
	// for a larger region and remove the parts we don't need.
	retries := 0
retry:
	p := uintptr(sysReserve(v, size+align, vmaName))
	switch {
	case p == 0:
		return nil, 0
	case p&(align-1) == 0:
		return unsafe.Pointer(p), size + align
	case GOOS == "windows":
		// On Windows we can't release pieces of a
		// reservation, so we release the whole thing and
		// re-reserve the aligned sub-region. This may race,
		// so we may have to try again.
		sysUnreserve(unsafe.Pointer(p), size+align)
		p = alignUp(p, align)
		p2 := sysReserve(unsafe.Pointer(p), size, vmaName)
		if p != uintptr(p2) {
			// Must have raced. Try again.
			sysUnreserve(p2, size)
			if retries++; retries == 100 {
				throw("failed to allocate aligned heap memory; too many retries")
			}
			goto retry
		}
		// Success.
		return p2, size
	default:
		// Trim off the unaligned parts.
		pAligned := alignUp(p, align)
		end := pAligned + size
		endLen := (p + size + align) - end

		// sysUnreserve does not allow unreserving a subset of the
		// region because LSAN does not allow unregistering a subset.
		// So we can't call sysUnreserve. Instead we simply unregister
		// the entire region from LSAN and re-register with the smaller
		// region before freeing the unecessary portions, which does
		// allow subsets of the region.
		if asanenabled {
			lsanunregisterrootregion(unsafe.Pointer(p), size+align)
			lsanregisterrootregion(unsafe.Pointer(pAligned), size)
		}
		sysFreeOS(unsafe.Pointer(p), pAligned-p)
		if endLen > 0 {
			sysFreeOS(unsafe.Pointer(end), endLen)
		}
		return unsafe.Pointer(pAligned), size
	}
}

// sysUnreserve transitions a memory region from Reserved to None.
//
// The size and start address must exactly match the size and returned address
// from sysReserve/sysReserveAligned. That is, sysUnreserve cannot be used to
// unreserve a subset of a memory region.
//
// Don't split the stack as this function may be invoked without a valid G,
// which prevents us from allocating more stack.
//
//go:nosplit
func sysUnreserve(v unsafe.Pointer, n uintptr) {
	// When using ASAN leak detection, the memory being freed is known by
	// the sanitizer. We need to unregister it so it's not accessed by it.
	//
	// lsanunregisterrootregion matches regions by start address and size,
	// so it is not possible to unregister a subset of the region. This is
	// why sysUnreserve requires the full region from sysReserve.
	if asanenabled {
		lsanunregisterrootregion(v, n)
	}

	sysFreeOS(v, n)
}

// sysMap transitions a memory region from Reserved to Prepared. It ensures the
// memory region can be efficiently transitioned to Ready.
//
// sysStat must be non-nil.
func sysMap(v unsafe.Pointer, n uintptr, sysStat *sysMemStat, vmaName string) {
	sysStat.add(int64(n))
	sysMapOS(v, n, vmaName)
}
