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

var is32bitOnly atomic.Bool

//go:nosplit
func futex(addr unsafe.Pointer, op int32, val uint32, ts *timespec, addr2 unsafe.Pointer, val3 uint32) int32 {
	if !is32bitOnly.Load() {
		ret := futex_time64(addr, op, val, ts, addr2, val3)
		// futex_time64 is only supported on Linux 5.0+
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
