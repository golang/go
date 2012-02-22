// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"os"
	"syscall"
	"testing"
	"time"
)

var icmpTests = []struct {
	net   string
	laddr string
	raddr string
	ipv6  bool // test with underlying AF_INET6 socket
}{
	{"ip4:icmp", "", "127.0.0.1", false},
	{"ip6:icmp", "", "::1", true},
}

func TestICMP(t *testing.T) {
	if os.Getuid() != 0 {
		t.Logf("test disabled; must be root")
		return
	}

	seqnum := 61455
	for _, tt := range icmpTests {
		if tt.ipv6 && !supportsIPv6 {
			continue
		}
		id := os.Getpid() & 0xffff
		seqnum++
		echo := newICMPEchoRequest(tt.net, id, seqnum, 128, []byte("Go Go Gadget Ping!!!"))
		exchangeICMPEcho(t, tt.net, tt.laddr, tt.raddr, echo)
	}
}

func exchangeICMPEcho(t *testing.T, net, laddr, raddr string, echo []byte) {
	c, err := ListenPacket(net, laddr)
	if err != nil {
		t.Errorf("ListenPacket(%q, %q) failed: %v", net, laddr, err)
		return
	}
	c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	defer c.Close()

	ra, err := ResolveIPAddr(net, raddr)
	if err != nil {
		t.Errorf("ResolveIPAddr(%q, %q) failed: %v", net, raddr, err)
		return
	}

	waitForReady := make(chan bool)
	go icmpEchoTransponder(t, net, raddr, waitForReady)
	<-waitForReady

	_, err = c.WriteTo(echo, ra)
	if err != nil {
		t.Errorf("WriteTo failed: %v", err)
		return
	}

	reply := make([]byte, 256)
	for {
		_, _, err := c.ReadFrom(reply)
		if err != nil {
			t.Errorf("ReadFrom failed: %v", err)
			return
		}
		switch c.(*IPConn).fd.family {
		case syscall.AF_INET:
			if reply[0] != ICMP4_ECHO_REPLY {
				continue
			}
		case syscall.AF_INET6:
			if reply[0] != ICMP6_ECHO_REPLY {
				continue
			}
		}
		xid, xseqnum := parseICMPEchoReply(echo)
		rid, rseqnum := parseICMPEchoReply(reply)
		if rid != xid || rseqnum != xseqnum {
			t.Errorf("ID = %v, Seqnum = %v, want ID = %v, Seqnum = %v", rid, rseqnum, xid, xseqnum)
			return
		}
		break
	}
}

func icmpEchoTransponder(t *testing.T, net, raddr string, waitForReady chan bool) {
	c, err := Dial(net, raddr)
	if err != nil {
		waitForReady <- true
		t.Errorf("Dial(%q, %q) failed: %v", net, raddr, err)
		return
	}
	c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	defer c.Close()
	waitForReady <- true

	echo := make([]byte, 256)
	var nr int
	for {
		nr, err = c.Read(echo)
		if err != nil {
			t.Errorf("Read failed: %v", err)
			return
		}
		switch c.(*IPConn).fd.family {
		case syscall.AF_INET:
			if echo[0] != ICMP4_ECHO_REQUEST {
				continue
			}
		case syscall.AF_INET6:
			if echo[0] != ICMP6_ECHO_REQUEST {
				continue
			}
		}
		break
	}

	switch c.(*IPConn).fd.family {
	case syscall.AF_INET:
		echo[0] = ICMP4_ECHO_REPLY
	case syscall.AF_INET6:
		echo[0] = ICMP6_ECHO_REPLY
	}

	_, err = c.Write(echo[:nr])
	if err != nil {
		t.Errorf("Write failed: %v", err)
		return
	}
}

const (
	ICMP4_ECHO_REQUEST = 8
	ICMP4_ECHO_REPLY   = 0
	ICMP6_ECHO_REQUEST = 128
	ICMP6_ECHO_REPLY   = 129
)

func newICMPEchoRequest(net string, id, seqnum, msglen int, filler []byte) []byte {
	afnet, _, _ := parseDialNetwork(net)
	switch afnet {
	case "ip4":
		return newICMPv4EchoRequest(id, seqnum, msglen, filler)
	case "ip6":
		return newICMPv6EchoRequest(id, seqnum, msglen, filler)
	}
	return nil
}

func newICMPv4EchoRequest(id, seqnum, msglen int, filler []byte) []byte {
	b := newICMPInfoMessage(id, seqnum, msglen, filler)
	b[0] = ICMP4_ECHO_REQUEST

	// calculate ICMP checksum
	cklen := len(b)
	s := uint32(0)
	for i := 0; i < cklen-1; i += 2 {
		s += uint32(b[i+1])<<8 | uint32(b[i])
	}
	if cklen&1 == 1 {
		s += uint32(b[cklen-1])
	}
	s = (s >> 16) + (s & 0xffff)
	s = s + (s >> 16)
	// place checksum back in header; using ^= avoids the
	// assumption the checksum bytes are zero
	b[2] ^= uint8(^s & 0xff)
	b[3] ^= uint8(^s >> 8)

	return b
}

func newICMPv6EchoRequest(id, seqnum, msglen int, filler []byte) []byte {
	b := newICMPInfoMessage(id, seqnum, msglen, filler)
	b[0] = ICMP6_ECHO_REQUEST
	return b
}

func newICMPInfoMessage(id, seqnum, msglen int, filler []byte) []byte {
	b := make([]byte, msglen)
	copy(b[8:], bytes.Repeat(filler, (msglen-8)/len(filler)+1))
	b[0] = 0                    // type
	b[1] = 0                    // code
	b[2] = 0                    // checksum
	b[3] = 0                    // checksum
	b[4] = uint8(id >> 8)       // identifier
	b[5] = uint8(id & 0xff)     // identifier
	b[6] = uint8(seqnum >> 8)   // sequence number
	b[7] = uint8(seqnum & 0xff) // sequence number
	return b
}

func parseICMPEchoReply(b []byte) (id, seqnum int) {
	id = int(b[4])<<8 | int(b[5])
	seqnum = int(b[6])<<8 | int(b[7])
	return
}
