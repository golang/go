// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var bloc uintptr
var memlock mutex

func memRound(p uintptr) uintptr {
	return (p + _PAGESIZE - 1) &^ (_PAGESIZE - 1)
}

func initBloc() {
	bloc = memRound(uintptr(unsafe.Pointer(&end)))
}

func sbrk(n uintptr) unsafe.Pointer {
	lock(&memlock)
	// Plan 9 sbrk from /sys/src/libc/9sys/sbrk.c
	bl := bloc
	n = memRound(n)
	if brk_(unsafe.Pointer(bl+n)) < 0 {
		unlock(&memlock)
		return nil
	}
	bloc += n
	unlock(&memlock)
	return unsafe.Pointer(bl)
}

func sysAlloc(n uintptr, stat *uint64) unsafe.Pointer {
	p := sbrk(n)
	if p != nil {
		xadd64(stat, int64(n))
	}
	return p
}

func sysFree(v unsafe.Pointer, n uintptr, stat *uint64) {
	xadd64(stat, -int64(n))
	lock(&memlock)
	// from tiny/mem.c
	// Push pointer back if this is a free
	// of the most recent sysAlloc.
	n = memRound(n)
	if bloc == uintptr(v)+n {
		bloc -= n
	}
	unlock(&memlock)
}

func sysUnused(v unsafe.Pointer, n uintptr) {
}

func sysUsed(v unsafe.Pointer, n uintptr) {
}

func sysMap(v unsafe.Pointer, n uintptr, reserved bool, stat *uint64) {
	// sysReserve has already allocated all heap memory,
	// but has not adjusted stats.
	xadd64(stat, int64(n))
}

func sysFault(v unsafe.Pointer, n uintptr) {
}

func sysReserve(v unsafe.Pointer, n uintptr, reserved *bool) unsafe.Pointer {
	*reserved = true
	return sbrk(n)
}
