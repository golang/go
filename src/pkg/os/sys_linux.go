// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linux-specific

package os

// Hostname returns the host name reported by the kernel.
func Hostname() (name string, err Error) {
	f, err := Open("/proc/sys/kernel/hostname")
	if err != nil {
		return "", err
	}
	defer f.Close()

	var buf [512]byte // Enough for a DNS name.
	n, err := f.Read(buf[0:])
	if err != nil {
		return "", err
	}

	if n > 0 && buf[n-1] == '\n' {
		n--
	}
	return string(buf[0:n]), nil
}
