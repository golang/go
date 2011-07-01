// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Sockets for Windows

package net

import (
	"syscall"
)

func setKernelSpecificSockopt(s syscall.Handle, f int) {
	// Allow reuse of recently-used addresses and ports.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)

	// Allow broadcast.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1)

	if f == syscall.AF_INET6 {
		// using ip, tcp, udp, etc.
		// allow both protocols even if the OS default is otherwise.
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, 0)
	}
}
