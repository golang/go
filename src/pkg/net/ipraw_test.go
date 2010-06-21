// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


// TODO(cw): ListenPacket test, Read() test, ipv6 test &
// Dial()/Listen() level tests

package net

import (
	"bytes"
	"flag"
	"os"
	"testing"
)

const ICMP_ECHO_REQUEST = 8
const ICMP_ECHO_REPLY = 0

// returns a suitable 'ping request' packet, with id & seq and a
// payload length of pktlen
func makePingRequest(id, seq, pktlen int, filler []byte) []byte {
	p := make([]byte, pktlen)
	copy(p[8:], bytes.Repeat(filler, (pktlen-8)/len(filler)+1))

	p[0] = ICMP_ECHO_REQUEST // type
	p[1] = 0                 // code
	p[2] = 0                 // cksum
	p[3] = 0                 // cksum
	p[4] = uint8(id >> 8)    // id
	p[5] = uint8(id & 0xff)  // id
	p[6] = uint8(seq >> 8)   // sequence
	p[7] = uint8(seq & 0xff) // sequence

	// calculate icmp checksum
	cklen := len(p)
	s := uint32(0)
	for i := 0; i < (cklen - 1); i += 2 {
		s += uint32(p[i+1])<<8 | uint32(p[i])
	}
	if cklen&1 == 1 {
		s += uint32(p[cklen-1])
	}
	s = (s >> 16) + (s & 0xffff)
	s = s + (s >> 16)

	// place checksum back in header; using ^= avoids the
	// assumption the checksum bytes are zero
	p[2] ^= uint8(^s & 0xff)
	p[3] ^= uint8(^s >> 8)

	return p
}

func parsePingReply(p []byte) (id, seq int) {
	id = int(p[4])<<8 | int(p[5])
	seq = int(p[6])<<8 | int(p[7])
	return
}

var srchost = flag.String("srchost", "", "Source of the ICMP ECHO request")
var dsthost = flag.String("dsthost", "localhost", "Destination for the ICMP ECHO request")

// test (raw) IP socket using ICMP
func TestICMP(t *testing.T) {
	if os.Getuid() != 0 {
		t.Logf("test disabled; must be root")
		return
	}

	var laddr *IPAddr
	if *srchost != "" {
		laddr, err := ResolveIPAddr(*srchost)
		if err != nil {
			t.Fatalf(`net.ResolveIPAddr("%v") = %v, %v`, *srchost, laddr, err)
		}
	}

	raddr, err := ResolveIPAddr(*dsthost)
	if err != nil {
		t.Fatalf(`net.ResolveIPAddr("%v") = %v, %v`, *dsthost, raddr, err)
	}

	c, err := ListenIP("ip4:icmp", laddr)
	if err != nil {
		t.Fatalf(`net.ListenIP("ip4:icmp", %v) = %v, %v`, *srchost, c, err)
	}

	sendid := os.Getpid() & 0xffff
	const sendseq = 61455
	const pingpktlen = 128
	sendpkt := makePingRequest(sendid, sendseq, pingpktlen, []byte("Go Go Gadget Ping!!!"))

	n, err := c.WriteToIP(sendpkt, raddr)
	if err != nil || n != pingpktlen {
		t.Fatalf(`net.WriteToIP(..., %v) = %v, %v`, raddr, n, err)
	}

	c.SetTimeout(100e6)
	resp := make([]byte, 1024)
	for {
		n, from, err := c.ReadFrom(resp)
		if err != nil {
			t.Fatalf(`ReadFrom(...) = %v, %v, %v`, n, from, err)
		}
		if resp[0] != ICMP_ECHO_REPLY {
			continue
		}
		rcvid, rcvseq := parsePingReply(resp)
		if rcvid != sendid || rcvseq != sendseq {
			t.Fatalf(`Ping reply saw id,seq=0x%x,0x%x (expected 0x%x, 0x%x)`, rcvid, rcvseq, sendid, sendseq)
		}
		return
	}
	t.Fatalf("saw no ping return")
}
