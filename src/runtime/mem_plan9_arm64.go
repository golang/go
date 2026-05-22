// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	plan9SgCexec          = 0x40
	plan9MemorySegmentLen = uintptr(4095 << 20)
)

var (
	plan9MemoryName = [...]byte{'m', 'e', 'm', 'o', 'r', 'y', 0}
	plan9UseBrk     bool
)

//go:noescape
func segattach(attr uintptr, name *byte, va unsafe.Pointer, length uintptr) unsafe.Pointer

func sbrk(n uintptr) unsafe.Pointer {
	n = memRound(n)
	if !plan9UseBrk {
		if blocMax == memRound(firstmoduledata.end) {
			attached := uintptr(segattach(plan9SgCexec, &plan9MemoryName[0], nil, plan9MemorySegmentLen))
			if attached != 0 && attached != ^uintptr(0) {
				end := attached + plan9MemorySegmentLen
				base := alignUp(attached, heapArenaBytes)
				if end < attached || base >= end {
					plan9UseBrk = true
				} else {
					bloc = base
					blocMax = end
				}
			} else {
				plan9UseBrk = true
			}
		}
		if !plan9UseBrk {
			bl := bloc
			if bl+n > blocMax || bl+n < bl {
				return nil
			}
			bloc += n
			return unsafe.Pointer(bl)
		}
	}

	// Fallback to the traditional Plan 9 BSS break.
	bl := bloc
	if bl+n > blocMax {
		if brk_(unsafe.Pointer(bl+n)) < 0 {
			return nil
		}
		blocMax = bl + n
	}
	bloc += n
	return unsafe.Pointer(bl)
}

func sysUnusedOSImpl(v unsafe.Pointer, n uintptr) {
	// Heap pages are not returned to the OS on plan9/arm64, matching the
	// other Plan 9 ports (386/arm/amd64), where sysUnusedOS is a no-op.
	//
	// Returning pages here via the segfree(2) syscall is unsafe under SMP:
	// segfree drops the physical page backing a virtual address, but unless the kernel
	// performs a cross-core TLB shootdown, another core can retain a stale
	// mapping and read/write the page after it has been recycled, corrupting
	// unrelated live allocations. This manifests as intermittent, SMP-
	// exacerbated heap corruption (e.g. panics deep inside the compiler while
	// building on 9front). Keep the segment reservation from segattach, but
	// do not release individual pages during scavenging.
}
