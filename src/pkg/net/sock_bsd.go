// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd

// Sockets for BSD variants

package net

import (
	"syscall"
)

func setKernelSpecificSockopt(s, f int) {
	// Allow reuse of recently-used addresses.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)

	// Allow reuse of recently-used ports.
	// This option is supported only in descendants of 4.4BSD,
	// to make an effective multicast application and an application
	// that requires quick draw possible.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEPORT, 1)

	// Allow broadcast.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1)

	if f == syscall.AF_INET6 {
		// using ip, tcp, udp, etc.
		// allow both protocols even if the OS default is otherwise.
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, 0)
	}
}
