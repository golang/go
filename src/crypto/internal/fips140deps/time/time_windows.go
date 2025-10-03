// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import "internal/syscall/windows"

// HighPrecisionNow returns a high-resolution timestamp suitable for measuring
// small time differences. It uses Windows' QueryPerformanceCounter.
//
// Its unit, epoch, and resolution are unspecified, and may change, but can be
// assumed to be sufficiently precise to measure time differences on the order
// of tens to hundreds of nanoseconds.
func HighPrecisionNow() int64 {
	return windows.QueryPerformanceCounter()
}
