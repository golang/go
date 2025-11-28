// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || arm || mips || mipsle || (gccgo && (ppc || s390)))

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

//go:noescape
func futex_time32(addr unsafe.Pointer, op int32, val uint32, ts *timespec32, addr2 unsafe.Pointer, val3 uint32) int32

//go:noescape
func futex_time64(addr unsafe.Pointer, op int32, val uint32, ts *timespec, addr2 unsafe.Pointer, val3 uint32) int32

var isFutexTime32bitOnly atomic.Bool

//go:nosplit
func futex(addr unsafe.Pointer, op int32, val uint32, ts *timespec, addr2 unsafe.Pointer, val3 uint32) int32 {
	if !isFutexTime32bitOnly.Load() {
		ret := futex_time64(addr, op, val, ts, addr2, val3)
		// futex_time64 is only supported on Linux 5.0+
		if ret != -_ENOSYS {
			return ret
		}
		isFutexTime32bitOnly.Store(true)
	}
	// Downgrade ts.
	var ts32 timespec32
	var pts32 *timespec32
	if ts != nil {
		ts32.setNsec(ts.tv_sec*1e9 + ts.tv_nsec)
		pts32 = &ts32
	}
	return futex_time32(addr, op, val, pts32, addr2, val3)
}

//go:noescape
func timer_settime32(timerid int32, flags int32, new, old *itimerspec32) int32

//go:noescape
func timer_settime64(timerid int32, flags int32, new, old *itimerspec) int32

var isSetTime32bitOnly atomic.Bool

//go:nosplit
func timer_settime(timerid int32, flags int32, new, old *itimerspec) int32 {
	if !isSetTime32bitOnly.Load() {
		ret := timer_settime64(timerid, flags, new, old)
		// timer_settime64 is only supported on Linux 5.0+
		if ret != -_ENOSYS {
			return ret
		}
		isSetTime32bitOnly.Store(true)
	}

	var newts, oldts itimerspec32
	var new32, old32 *itimerspec32

	if new != nil {
		newts.it_interval.setNsec(new.it_interval.tv_sec*1e9 + new.it_interval.tv_nsec)
		newts.it_value.setNsec(new.it_value.tv_sec*1e9 + new.it_value.tv_nsec)
		new32 = &newts
	}

	if old != nil {
		oldts.it_interval.setNsec(old.it_interval.tv_sec*1e9 + old.it_interval.tv_nsec)
		oldts.it_value.setNsec(old.it_value.tv_sec*1e9 + old.it_value.tv_nsec)
		old32 = &oldts
	}

	// Fall back to 32-bit timer
	return timer_settime32(timerid, flags, new32, old32)
}
