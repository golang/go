// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Socket options for Linux

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
	return nil
}
