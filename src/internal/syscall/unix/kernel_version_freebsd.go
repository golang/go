// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"sync"
	"syscall"
)

// KernelVersion returns major and minor kernel version numbers
// parsed from the syscall.Sysctl("kern.osrelease")'s value,
// or (0, 0) if the version can't be obtained or parsed.
func KernelVersion() (major, minor int) {
	release, err := syscall.Sysctl("kern.osrelease")
	if err != nil {
		return 0, 0
	}

	parseNext := func() (n int) {
		for i, c := range release {
			if c == '.' {
				release = release[i+1:]
				return
			}
			if '0' <= c && c <= '9' {
				n = n*10 + int(c-'0')
			}
		}
		release = ""
		return
	}

	major = parseNext()
	minor = parseNext()

	return
}

// SupportCopyFileRange reports whether the kernel supports the copy_file_range(2).
// This function will examine both the kernel version and the availability of the system call.
var SupportCopyFileRange = sync.OnceValue(func() bool {
	// The copy_file_range() function first appeared in FreeBSD 13.0.
	if !KernelVersionGE(13, 0) {
		return false
	}
	_, err := CopyFileRange(0, nil, 0, nil, 0, 0)
	return err != syscall.ENOSYS
})
