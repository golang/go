// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl solaris

package unix

import "syscall"

func getsockname(s int, addr []byte) error {
	return syscall.EOPNOTSUPP
}

func getpeername(s int, addr []byte) error {
	return syscall.EOPNOTSUPP
}

func recvfrom(s int, b []byte, flags int, from []byte) (int, error) {
	return 0, syscall.EOPNOTSUPP
}

func sendto(s int, b []byte, flags int, to []byte) (int, error) {
	return 0, syscall.EOPNOTSUPP
}
