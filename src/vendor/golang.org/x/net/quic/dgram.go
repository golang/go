// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"net/netip"
	"sync"
)

type datagram struct {
	b         []byte
	localAddr netip.AddrPort
	peerAddr  netip.AddrPort
	ecn       ecnBits
}

// Explicit Congestion Notification bits.
//
// https://www.rfc-editor.org/rfc/rfc3168.html#section-5
type ecnBits byte

const (
	ecnMask   = 0b000000_11
	ecnNotECT = 0b000000_00
	ecnECT1   = 0b000000_01
	ecnECT0   = 0b000000_10
	ecnCE     = 0b000000_11
)

var datagramPool = sync.Pool{
	New: func() any {
		return &datagram{
			b: make([]byte, maxUDPPayloadSize),
		}
	},
}

func newDatagram() *datagram {
	m := datagramPool.Get().(*datagram)
	*m = datagram{
		b: m.b[:cap(m.b)],
	}
	return m
}

func (m *datagram) recycle() {
	if cap(m.b) != maxUDPPayloadSize {
		return
	}
	datagramPool.Put(m)
}
