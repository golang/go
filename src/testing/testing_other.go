// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

package testing

import "time"

// isWindowsRetryable reports whether err is a Windows error code
// that may be fixed by retrying a failed filesystem operation.
func isWindowsRetryable(err error) bool {
	return false
}

// highPrecisionTime represents a single point in time.
// On all systems except Windows, using time.Time is fine.
type highPrecisionTime struct {
	now time.Time
}

// highPrecisionTimeNow returns high precision time for benchmarking.
func highPrecisionTimeNow() highPrecisionTime {
	return highPrecisionTime{now: time.Now()}
}

// highPrecisionTimeSince returns duration since b.
func highPrecisionTimeSince(b highPrecisionTime) time.Duration {
	return time.Since(b.now)
}
