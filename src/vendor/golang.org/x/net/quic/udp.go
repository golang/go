// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "net/netip"

// Per-plaform consts describing support for various features.
//
// const udpECNSupport indicates whether the platform supports setting
// the ECN (Explicit Congestion Notification) IP header bits.
//
// const udpInvalidLocalAddrIsError indicates whether sending a packet
// from an local address not associated with the system is an error.
// For example, assuming 127.0.0.2 is not a local address, does sending
// from it (using IP_PKTINFO or some other such feature) result in an error?

// unmapAddrPort returns a with any IPv4-mapped IPv6 address prefix removed.
func unmapAddrPort(a netip.AddrPort) netip.AddrPort {
	if a.Addr().Is4In6() {
		return netip.AddrPortFrom(
			a.Addr().Unmap(),
			a.Port(),
		)
	}
	return a
}
