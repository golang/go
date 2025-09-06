// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Same as os_linux_futex.go, but with a fallback in case 32-bit compat is not
// configured. The Linux kernel introduced a new syscall for 32-bit systems to
// solve the year 2038 issue with 32-bit timestamps. In order to still be able
// to use the older syscall, the `CONFIG_COMPAT_32BIT_TIME` option needs to be
// enabled.

//go:build linux && (386 || arm || mips || mipsle || ppc)

package runtime

import (
	"internal/runtime/atomic"
	"unsafe"
)

//go:noescape
func futex_time32(addr unsafe.Pointer, op int32, val uint32, ts *timespec32, addr2 unsafe.Pointer, val3 uint32) int32

//go:noescape
func futex_time64(addr unsafe.Pointer, op int32, val uint32, ts *timespec, addr2 unsafe.Pointer, val3 uint32) int32

// Before Linux 5.0, there was only the 32-bit syscall.
var is32bitOnly atomic.Bool

//go:nosplit
func futex(addr unsafe.Pointer, op int32, val uint32, ts *timespec, addr2 unsafe.Pointer, val3 uint32) int32 {
	// If the newer futex_time64 is not available, fall back to old futex syscall.
	if !is32bitOnly.Load() {
		ret := futex_time64(addr, op, val, ts, addr2, val3)
		if ret != -_ENOSYS {
			return ret
		}
		is32bitOnly.Store(true)
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
