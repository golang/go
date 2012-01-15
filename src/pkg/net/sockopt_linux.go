// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Socket options for Linux

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
}
