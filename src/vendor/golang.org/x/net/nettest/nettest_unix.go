// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package nettest

import "syscall"

func supportsRawSocket() bool {
	for _, af := range []int{syscall.AF_INET, syscall.AF_INET6} {
		s, err := syscall.Socket(af, syscall.SOCK_RAW, 0)
		if err != nil {
			continue
		}
		syscall.Close(s)
		return true
	}
	return false
}
