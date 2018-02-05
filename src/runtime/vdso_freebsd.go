// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

const _VDSO_TH_NUM = 4 // defined in <sys/vdso.h> #ifdef _KERNEL

var timekeepSharedPage *vdsoTimekeep

//go:nosplit
func (bt bintime) Add(bt2 bintime) bintime {
	u := bt.frac
	bt.frac += bt2.frac
	if u > bt.frac {
		bt.sec++
	}
	bt.sec += bt2.sec
	return bt
}

//go:nosplit
func (bt bintime) AddX(x uint64) bintime {
	u := bt.frac
	bt.frac += x
	if u > bt.frac {
		bt.sec++
	}
	return bt
}

var binuptimeDummy uint32

// based on /usr/src/lib/libc/sys/__vdso_gettimeofday.c
//
//go:nosplit
func binuptime(abs bool) (bintime, bool) {
	var bt bintime
	timehands := (*[_VDSO_TH_NUM]vdsoTimehands)(add(unsafe.Pointer(timekeepSharedPage), vdsoTimekeepSize))
	for {
		if timekeepSharedPage.enabled == 0 {
			return bt, false
		}

		curr := atomic.Load(&timekeepSharedPage.current) // atomic_load_acq_32
		th := &timehands[curr]
		gen := atomic.Load(&th.gen) // atomic_load_acq_32
		bt = th.offset

		if tc, ok := th.getTimecounter(); !ok {
			return bt, false
		} else {
			delta := (tc - th.offset_count) & th.counter_mask
			bt = bt.AddX(th.scale * uint64(delta))
		}
		if abs {
			bt = bt.Add(th.boottime)
		}

		atomic.Load(&binuptimeDummy) // atomic_thread_fence_acq()
		if curr == timekeepSharedPage.current && gen != 0 && gen == th.gen {
			break
		}
	}
	return bt, true
}

//go:nosplit
func vdsoClockGettime(clockID int32) (bintime, bool) {
	if timekeepSharedPage == nil || timekeepSharedPage.ver != _VDSO_TK_VER_CURR {
		return bintime{}, false
	}
	abs := false
	switch clockID {
	case _CLOCK_MONOTONIC:
		/* ok */
	case _CLOCK_REALTIME:
		abs = true
	default:
		return bintime{}, false
	}

	return binuptime(abs)
}

func fallback_nanotime() int64
func fallback_walltime() (sec int64, nsec int32)

//go:nosplit
func nanotime() int64 {
	bt, ok := vdsoClockGettime(_CLOCK_MONOTONIC)
	if !ok {
		return fallback_nanotime()
	}
	return int64((1e9 * uint64(bt.sec)) + ((1e9 * uint64(bt.frac>>32)) >> 32))
}

func walltime() (sec int64, nsec int32) {
	bt, ok := vdsoClockGettime(_CLOCK_REALTIME)
	if !ok {
		return fallback_walltime()
	}
	return int64(bt.sec), int32((1e9 * uint64(bt.frac>>32)) >> 32)
}
