// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package testing

import (
	"errors"
	"internal/syscall/windows"
	"math/bits"
	"syscall"
	"time"
)

// isWindowsRetryable reports whether err is a Windows error code
// that may be fixed by retrying a failed filesystem operation.
func isWindowsRetryable(err error) bool {
	for {
		unwrapped := errors.Unwrap(err)
		if unwrapped == nil {
			break
		}
		err = unwrapped
	}
	if err == syscall.ERROR_ACCESS_DENIED {
		return true // Observed in https://go.dev/issue/50051.
	}
	if err == windows.ERROR_SHARING_VIOLATION {
		return true // Observed in https://go.dev/issue/51442.
	}
	return false
}

// highPrecisionTime represents a single point in time with query performance counter.
// time.Time on Windows has low system granularity, which is not suitable for
// measuring short time intervals.
//
// TODO: If Windows runtime implements high resolution timing then highPrecisionTime
// can be removed.
type highPrecisionTime struct {
	now int64
}

// highPrecisionTimeNow returns high precision time for benchmarking.
func highPrecisionTimeNow() highPrecisionTime {
	var t highPrecisionTime
	// This should always succeed for Windows XP and above.
	t.now = windows.QueryPerformanceCounter()
	return t
}

func (a highPrecisionTime) sub(b highPrecisionTime) time.Duration {
	delta := a.now - b.now

	if queryPerformanceFrequency == 0 {
		queryPerformanceFrequency = windows.QueryPerformanceFrequency()
	}
	hi, lo := bits.Mul64(uint64(delta), uint64(time.Second)/uint64(time.Nanosecond))
	quo, _ := bits.Div64(hi, lo, uint64(queryPerformanceFrequency))
	return time.Duration(quo)
}

var queryPerformanceFrequency int64

// highPrecisionTimeSince returns duration since a.
func highPrecisionTimeSince(a highPrecisionTime) time.Duration {
	return highPrecisionTimeNow().sub(a)
}
