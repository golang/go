// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd
// +build 386 amd64

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
)

var (
	hpetDevMap  [_HPET_DEV_MAP_MAX]uintptr
	hpetDevPath = [_HPET_DEV_MAP_MAX][11]byte{
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '0', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '1', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '2', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '3', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '4', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '5', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '6', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '7', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '8', 0},
		{'/', 'd', 'e', 'v', '/', 'h', 'p', 'e', 't', '9', 0},
	}
)

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
		fd := open(&hpetDevPath[idx][0], 0 /* O_RDONLY */, 0)
		if fd < 0 {
			atomic.Casuintptr(&hpetDevMap[idx], 0, ^uintptr(0))
			return 0, false
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
		p = atomic.Loaduintptr(&hpetDevMap[idx])
	}
	if p == ^uintptr(0) {
		return 0, false
	}
	return *(*uint32)(unsafe.Pointer(p + _HPET_MAIN_COUNTER)), true
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
