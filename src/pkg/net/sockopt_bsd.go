// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

// Socket options for BSD variants

package net

import (
	"syscall"
)

func setDefaultSockopts(s, f, p int) {
	switch f {
	case syscall.AF_INET6:
		// Allow both IP versions even if the OS default is otherwise.
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, 0)
	}

	if f == syscall.AF_UNIX || p == syscall.IPPROTO_TCP {
		// Allow reuse of recently-used addresses.
		syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)

		// Allow reuse of recently-used ports.
		// This option is supported only in descendants of 4.4BSD,
		// to make an effective multicast application and an application
		// that requires quick draw possible.
		syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEPORT, 1)
	}

	// Allow broadcast.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1)
}

func setDefaultMulticastSockopts(fd *netFD) {
	fd.incref()
	defer fd.decref()
	// Allow multicast UDP and raw IP datagram sockets to listen
	// concurrently across multiple listeners.
	syscall.SetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)
	syscall.SetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_REUSEPORT, 1)
}
