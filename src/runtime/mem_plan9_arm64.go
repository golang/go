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

//go:noescape
func segfree(addr unsafe.Pointer, length uintptr) int32

func sbrk(n uintptr) unsafe.Pointer {
	n = memRound(n)
	if !plan9UseBrk {
		if blocMax == memRound(firstmoduledata.end) {
			base := uintptr(segattach(plan9SgCexec, &plan9MemoryName[0], nil, plan9MemorySegmentLen))
			if base != 0 && base != ^uintptr(0) {
				bloc = base
				blocMax = base + plan9MemorySegmentLen
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
	if !plan9UseBrk {
		segfree(v, n)
	}
}
