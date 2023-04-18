// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd && (386 || amd64)

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

const (
	_VDSO_TH_ALGO_X86_TSC  = 1
	_VDSO_TH_ALGO_X86_HPET = 2
)

const (
	_HPET_DEV_MAP_MAX  = 10
	_HPET_MAIN_COUNTER = 0xf0 /* Main counter register */

	hpetDevPath = "/dev/hpetX\x00"
)

var hpetDevMap [_HPET_DEV_MAP_MAX]uintptr

//go:nosplit
func (th *vdsoTimehands) getTSCTimecounter() uint32 {
	tsc := cputicks()
	if th.x86_shift > 0 {
		tsc >>= th.x86_shift
	}
	return uint32(tsc)
}

//go:nosplit
func (th *vdsoTimehands) getHPETTimecounter() (uint32, bool) {
	idx := int(th.x86_hpet_idx)
	if idx >= len(hpetDevMap) {
		return 0, false
	}

	p := atomic.Loaduintptr(&hpetDevMap[idx])
	if p == 0 {
		systemstack(func() { initHPETTimecounter(idx) })
		p = atomic.Loaduintptr(&hpetDevMap[idx])
	}
	if p == ^uintptr(0) {
		return 0, false
	}
	return *(*uint32)(unsafe.Pointer(p + _HPET_MAIN_COUNTER)), true
}

//go:systemstack
func initHPETTimecounter(idx int) {
	const digits = "0123456789"

	var devPath [len(hpetDevPath)]byte
	copy(devPath[:], hpetDevPath)
	devPath[9] = digits[idx]

	fd := open(&devPath[0], 0 /* O_RDONLY */ |_O_CLOEXEC, 0)
	if fd < 0 {
		atomic.Casuintptr(&hpetDevMap[idx], 0, ^uintptr(0))
		return
	}

	addr, mmapErr := mmap(nil, physPageSize, _PROT_READ, _MAP_SHARED, fd, 0)
	closefd(fd)
	newP := uintptr(addr)
	if mmapErr != 0 {
		newP = ^uintptr(0)
	}
	if !atomic.Casuintptr(&hpetDevMap[idx], 0, newP) && mmapErr == 0 {
		munmap(addr, physPageSize)
	}
}

//go:nosplit
func (th *vdsoTimehands) getTimecounter() (uint32, bool) {
	switch th.algo {
	case _VDSO_TH_ALGO_X86_TSC:
		return th.getTSCTimecounter(), true
	case _VDSO_TH_ALGO_X86_HPET:
		return th.getHPETTimecounter()
	default:
		return 0, false
	}
}
