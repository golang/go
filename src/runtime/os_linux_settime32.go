// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || arm || mips || mipsle)

package runtime

import "internal/runtime/atomic"

var timer32bitOnly atomic.Bool

//go:noescape
func timer_settime32(timerid int32, flags int32, new, old *itimerspec32) int32

//go:noescape
func timer_settime64(timerid int32, flags int32, new, old *itimerspec) int32

//go:nosplit
func timer_settime(timerid int32, flags int32, new, old *itimerspec) int32 {
	if !timer32bitOnly.Load() {
		ret := timer_settime64(timerid, flags, new, old)
		// timer_settime64 is only supported on Linux 5.0+
		if ret != -_ENOSYS {
			return ret
		}
		timer32bitOnly.Store(true)
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
