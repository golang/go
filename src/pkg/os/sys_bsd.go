// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd openbsd

// os code shared between *BSD systems including OS X (Darwin)
// and FreeBSD.

package os

import "syscall"

func Hostname() (name string, err Error) {
	var errno int
	name, errno = syscall.Sysctl("kern.hostname")
	if errno != 0 {
		return "", NewSyscallError("sysctl kern.hostname", errno)
	}
	return name, nil
}
