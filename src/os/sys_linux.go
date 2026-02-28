// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"runtime"
	"syscall"
)

func hostname() (name string, err error) {
	// Try uname first, as it's only one system call and reading
	// from /proc is not allowed on Android.
	var un syscall.Utsname
	err = syscall.Uname(&un)

	var buf [512]byte // Enough for a DNS name.
	for i, b := range un.Nodename[:] {
		buf[i] = uint8(b)
		if b == 0 {
			name = string(buf[:i])
			break
		}
	}
	// If we got a name and it's not potentially truncated
	// (Nodename is 65 bytes), return it.
	if err == nil && len(name) > 0 && len(name) < 64 {
		return name, nil
	}
	if runtime.GOOS == "android" {
		if name != "" {
			return name, nil
		}
		return "localhost", nil
	}

	f, err := Open("/proc/sys/kernel/hostname")
	if err != nil {
		return "", err
	}
	defer f.Close()

	n, err := f.Read(buf[:])
	if err != nil {
		return "", err
	}

	if n > 0 && buf[n-1] == '\n' {
		n--
	}
	return string(buf[:n]), nil
}
