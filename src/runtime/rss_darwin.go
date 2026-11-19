// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

func sysRSSInit() {}

// sysRSS returns the current resident set size of this process in bytes,
// including pages shared with other processes.
func sysRSS() (uint64, bool) {
	var info machTaskBasicInfo
	if task_info(_MACH_TASK_BASIC_INFO, &info, _MACH_TASK_BASIC_INFO_COUNT) != 0 {
		return 0, false
	}
	return info.resident_size, true
}
