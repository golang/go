// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Sockets for Windows

package net

import (
	"syscall"
)

func setKernelSpecificSockopt(s syscall.Handle, f int) {
	// Windows will reuse recently-used addresses by default.
	// SO_REUSEADDR should not be used here, as it allows
	// a socket to forcibly bind to a port in use by another socket.
	// This could lead to a non-deterministic behavior, where
	// connection requests over the port cannot be guaranteed
	// to be handled by the correct socket.

	// Allow broadcast.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1)

	if f == syscall.AF_INET6 {
		// using ip, tcp, udp, etc.
		// allow both protocols even if the OS default is otherwise.
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, 0)
	}
}
