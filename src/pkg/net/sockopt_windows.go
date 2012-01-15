// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Socket options for Windows

package net

import (
	"syscall"
)

func setDefaultSockopts(s syscall.Handle, f, p int) {
	switch f {
	case syscall.AF_INET6:
		// Allow both IP versions even if the OS default is otherwise.
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, 0)
	}

	// Windows will reuse recently-used addresses by default.
	// SO_REUSEADDR should not be used here, as it allows
	// a socket to forcibly bind to a port in use by another socket.
	// This could lead to a non-deterministic behavior, where
	// connection requests over the port cannot be guaranteed
	// to be handled by the correct socket.

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
