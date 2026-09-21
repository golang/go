// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"runtime"
	"syscall"
)

func hostname() (string, error) {
	// Try uname first, as it's only one system call.
	var un syscall.Utsname
	if err := syscall.Uname(&un); err == nil && un.Nodename[0] != 0 {
		var buf [len(un.Nodename)]byte
		for i, b := range un.Nodename[:] {
			buf[i] = byte(b) // b is signed on some platforms
			if b == 0 {
				return string(buf[:i]), nil
			}
		}
	}

	// Fall back to /proc, except on Android where that is not allowed.
	if runtime.GOOS == "android" {
		return "localhost", nil
	}

	f, err := Open("/proc/sys/kernel/hostname")
	if err != nil {
		return "", err
	}
	defer f.Close()

	var buf [512]byte // Enough for a DNS name.
	n, err := f.Read(buf[:])
	if err != nil {
		return "", err
	}

	if n > 0 && buf[n-1] == '\n' {
		n--
	}
	return string(buf[:n]), nil
}
