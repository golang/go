// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"errors"
	"internal/poll"
	"sync"
	"syscall"
)

var (
	mptcpOnce      sync.Once
	mptcpAvailable bool
)

// These constants aren't in the syscall package, which is frozen
const (
	_IPPROTO_MPTCP = 0x106
)

func supportsMultipathTCP() bool {
	mptcpOnce.Do(initMPTCPavailable)
	return mptcpAvailable
}

// Check that MPTCP is supported by attemting to create an MPTCP socket and by
// looking at the returned error if any.
func initMPTCPavailable() {
	s, err := sysSocket(syscall.AF_INET, syscall.SOCK_STREAM, _IPPROTO_MPTCP)
	switch {
	case errors.Is(err, syscall.EPROTONOSUPPORT): // Not supported: >= v5.6
	case errors.Is(err, syscall.EINVAL): // Not supported: < v5.6
	case err == nil: // Supported and no error
		poll.CloseFunc(s)
		fallthrough
	default:
		// another error: MPTCP was not available but it might be later
		mptcpAvailable = true
	}
}

func (sd *sysDialer) dialMPTCP(ctx context.Context, laddr, raddr *TCPAddr) (*TCPConn, error) {
	// Fallback to dialTCP if Multipath TCP isn't supported on this operating system.
	if !supportsMultipathTCP() {
		return sd.dialTCP(ctx, laddr, raddr)
	}

	return sd.doDialTCPProto(ctx, laddr, raddr, _IPPROTO_MPTCP)
}
