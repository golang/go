// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

// Socket options for BSD variants

package net

import (
	"os"
	"syscall"
)

func setDefaultSockopts(s, f, t int) error {
	switch f {
	case syscall.AF_INET6:
		// Allow both IP versions even if the OS default is otherwise.
		// Note that some operating systems never admit this option.
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, 0)
	}

	if f == syscall.AF_UNIX ||
		(f == syscall.AF_INET || f == syscall.AF_INET6) && t == syscall.SOCK_STREAM {
		// Allow reuse of recently-used addresses.
		err := syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)
		if err != nil {
			return os.NewSyscallError("setsockopt", err)
		}

		// Allow reuse of recently-used ports.
		// This option is supported only in descendants of 4.4BSD,
		// to make an effective multicast application and an application
		// that requires quick draw possible.
		err = syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEPORT, 1)
		if err != nil {
			return os.NewSyscallError("setsockopt", err)
		}
	}

	// Allow broadcast.
	err := syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}

	return nil
}

func setDefaultMulticastSockopts(s int) error {
	// Allow multicast UDP and raw IP datagram sockets to listen
	// concurrently across multiple listeners.
	err := syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	err = syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEPORT, 1)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}
