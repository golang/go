// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package quic

import (
	"syscall"
)

const (
	ip_recvtos       = syscall.IP_RECVTOS
	ipv6_recvpktinfo = syscall.IPV6_RECVPKTINFO
	ipv6_pktinfo     = syscall.IPV6_PKTINFO
)

// See udp.go.
const (
	udpECNSupport              = true
	udpInvalidLocalAddrIsError = false
)

// The IP_TOS socket option is a single byte containing the IP TOS field.
// The low two bits are the ECN field.

func parseIPTOS(b []byte) (ecnBits, bool) {
	if len(b) != 1 {
		return 0, false
	}
	return ecnBits(b[0] & ecnMask), true
}

func appendCmsgECNv4(b []byte, ecn ecnBits) []byte {
	b, data := appendCmsg(b, syscall.IPPROTO_IP, syscall.IP_TOS, 1)
	data[0] = byte(ecn)
	return b
}
