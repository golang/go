// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || arm || mips || mipsle || (gccgo && (ppc || s390)))

package runtime

import (
	"unsafe"
)

// configure64bitsTimeOn32BitsArchitectures decides whether to use the
// 64-bit time variants of futex and timer_settime on 32-bit Linux.
//
// The choice is normally made by parsing the kernel release string
// from uname, because probing with -ENOSYS misbehaves on some
// kernels: Android 8.0-10 (API 26-29) has a seccomp filter that kills
// the process on unknown syscalls instead of returning -ENOSYS, and
// some older Synology kernels reuse the new syscall numbers for
// unrelated vendor syscalls, which silently runs the wrong thing.
//
// If the kernel release string can't be parsed, fall back to probing
// futex_time64 with a no-op FUTEX_WAKE: the kernels that motivated
// the version check above all report parseable uname strings, so
// reaching the probing fallback generally means we are on neither,
// and probing is the safest available signal.
func configure64bitsTimeOn32BitsArchitectures() {
	if kv, ok := getKernelVersion(); ok {
		use64bitsTimeOn32bits = kv.GE(5, 1)
		return
	}
	use64bitsTimeOn32bits = probeFutexTime64()
}

// probeFutexTime64 reports whether the futex_time64 syscall is
// available on the running kernel. It issues a FUTEX_WAKE_PRIVATE
// with a wake count of 0, which is a no-op if the syscall exists and
// returns -ENOSYS otherwise. timer_settime64 was added to the kernel
// in the same release (Linux 5.1), so a single probe covers both.
func probeFutexTime64() bool {
	var word uint32
	ret := futex_time64(unsafe.Pointer(&word), _FUTEX_WAKE_PRIVATE, 0, nil, nil, 0)
	return ret != -_ENOSYS
}

//go:noescape
func futex_time32(addr unsafe.Pointer, op int32, val uint32, ts *timespec32, addr2 unsafe.Pointer, val3 uint32) int32

//go:noescape
func futex_time64(addr unsafe.Pointer, op int32, val uint32, ts *timespec, addr2 unsafe.Pointer, val3 uint32) int32

var use64bitsTimeOn32bits bool

//go:nosplit
func futex(addr unsafe.Pointer, op int32, val uint32, ts *timespec, addr2 unsafe.Pointer, val3 uint32) int32 {
	if use64bitsTimeOn32bits {
		return futex_time64(addr, op, val, ts, addr2, val3)
	}

	var ts32 timespec32
	var pts32 *timespec32
	if ts != nil {
		ts32 = ts.mustDowncastToTimespec32()
		pts32 = &ts32
	}
	return futex_time32(addr, op, val, pts32, addr2, val3)
}

//go:noescape
func timer_settime32(timerid int32, flags int32, new, old *itimerspec32) int32

//go:noescape
func timer_settime64(timerid int32, flags int32, new, old *itimerspec) int32

//go:nosplit
func timer_settime(timerid int32, flags int32, new, old *itimerspec) int32 {
	if use64bitsTimeOn32bits {
		return timer_settime64(timerid, flags, new, old)
	}

	var newts, oldts itimerspec32
	var new32, old32 *itimerspec32

	if new != nil {
		newts.it_interval = new.it_interval.mustDowncastToTimespec32()
		newts.it_value = new.it_value.mustDowncastToTimespec32()
		new32 = &newts
	}

	if old != nil {
		oldts.it_interval = old.it_interval.mustDowncastToTimespec32()
		oldts.it_value = old.it_value.mustDowncastToTimespec32()
		old32 = &oldts
	}

	return timer_settime32(timerid, flags, new32, old32)
}

// mustDowncastToTimespec32 converts a 64-bit timespec to a 32-bit
// timespec and throws if the value cannot be represented in 32 bits.
//
//go:nosplit
func (ts timespec) mustDowncastToTimespec32() (r timespec32) {
	r.tv_sec = int32(ts.tv_sec)
	if ts.tv_sec != int64(r.tv_sec) {
		throw("timespec64 value cannot be represented in timespec32")
	}
	r.tv_nsec = ts.tv_nsec
	return
}
