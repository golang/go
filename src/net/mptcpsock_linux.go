// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"errors"
	"internal/poll"
	"internal/syscall/unix"
	"sync"
	"syscall"
)

var (
	mptcpOnce      sync.Once
	mptcpAvailable bool
	hasSOLMPTCP    bool // only valid if mptcpAvailable is true
)

// These constants aren't in the syscall package, which is frozen
const (
	_IPPROTO_MPTCP = 0x106
	_SOL_MPTCP     = 0x11c
	_MPTCP_INFO    = 0x1
)

func supportsMultipathTCP() bool {
	mptcpOnce.Do(initMPTCPavailable)
	return mptcpAvailable
}

// Check that MPTCP is supported by attempting to create an MPTCP socket and by
// looking at the returned error if any.
func initMPTCPavailable() {
	family := syscall.AF_INET
	if !supportsIPv4() {
		family = syscall.AF_INET6
	}
	s, err := sysSocket(family, syscall.SOCK_STREAM, _IPPROTO_MPTCP)

	switch {
	case errors.Is(err, syscall.EPROTONOSUPPORT): // Not supported: >= v5.6
		return
	case errors.Is(err, syscall.EINVAL): // Not supported: < v5.6
		return
	case err == nil: // Supported and no error
		poll.CloseFunc(s)
		fallthrough
	default:
		// another error: MPTCP was not available but it might be later
		mptcpAvailable = true
	}

	// SOL_MPTCP only supported from kernel 5.16.
	hasSOLMPTCP = unix.KernelVersionGE(5, 16)
}

func (sd *sysDialer) dialMPTCP(ctx context.Context, laddr, raddr *TCPAddr) (*TCPConn, error) {
	if supportsMultipathTCP() {
		if conn, err := sd.doDialTCPProto(ctx, laddr, raddr, _IPPROTO_MPTCP); err == nil {
			return conn, nil
		}
	}

	// Fallback to dialTCP if Multipath TCP isn't supported on this operating
	// system. But also fallback in case of any error with MPTCP.
	//
	// Possible MPTCP specific error: ENOPROTOOPT (sysctl net.mptcp.enabled=0)
	// But just in case MPTCP is blocked differently (SELinux, etc.), just
	// retry with "plain" TCP.
	return sd.dialTCP(ctx, laddr, raddr)
}

func (sl *sysListener) listenMPTCP(ctx context.Context, laddr *TCPAddr) (*TCPListener, error) {
	if supportsMultipathTCP() {
		if dial, err := sl.listenTCPProto(ctx, laddr, _IPPROTO_MPTCP); err == nil {
			return dial, nil
		}
	}

	// Fallback to listenTCP if Multipath TCP isn't supported on this operating
	// system. But also fallback in case of any error with MPTCP.
	//
	// Possible MPTCP specific error: ENOPROTOOPT (sysctl net.mptcp.enabled=0)
	// But just in case MPTCP is blocked differently (SELinux, etc.), just
	// retry with "plain" TCP.
	return sl.listenTCP(ctx, laddr)
}

// hasFallenBack reports whether the MPTCP connection has fallen back to "plain"
// TCP.
//
// A connection can fallback to TCP for different reasons, e.g. the other peer
// doesn't support it, a middle box "accidentally" drops the option, etc.
//
// If the MPTCP protocol has not been requested when creating the socket, this
// method will return true: MPTCP is not being used.
//
// Kernel >= 5.16 returns EOPNOTSUPP/ENOPROTOOPT in case of fallback.
// Older kernels will always return them even if MPTCP is used: not usable.
func hasFallenBack(fd *netFD) bool {
	_, err := fd.pfd.GetsockoptInt(_SOL_MPTCP, _MPTCP_INFO)

	// 2 expected errors in case of fallback depending on the address family
	//   - AF_INET:  EOPNOTSUPP
	//   - AF_INET6: ENOPROTOOPT
	return err == syscall.EOPNOTSUPP || err == syscall.ENOPROTOOPT
}

// isUsingMPTCPProto reports whether the socket protocol is MPTCP.
//
// Compared to hasFallenBack method, here only the socket protocol being used is
// checked: it can be MPTCP but it doesn't mean MPTCP is used on the wire, maybe
// a fallback to TCP has been done.
func isUsingMPTCPProto(fd *netFD) bool {
	proto, _ := fd.pfd.GetsockoptInt(syscall.SOL_SOCKET, syscall.SO_PROTOCOL)

	return proto == _IPPROTO_MPTCP
}

// isUsingMultipathTCP reports whether MPTCP is still being used.
//
// Please look at the description of hasFallenBack (kernel >=5.16) and
// isUsingMPTCPProto methods for more details about what is being checked here.
func isUsingMultipathTCP(fd *netFD) bool {
	if !supportsMultipathTCP() {
		return false
	}

	if hasSOLMPTCP {
		return !hasFallenBack(fd)
	}

	return isUsingMPTCPProto(fd)
}
