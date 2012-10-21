// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net_test

import (
	"bytes"
	"net"
	"os"
	"runtime"
	"testing"
	"time"
)

func TestUDPConnSpecificMethods(t *testing.T) {
	la, err := net.ResolveUDPAddr("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.ResolveUDPAddr failed: %v", err)
	}
	c, err := net.ListenUDP("udp4", la)
	if err != nil {
		t.Fatalf("net.ListenUDP failed: %v", err)
	}
	c.File()
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetReadBuffer(2048)
	c.SetWriteBuffer(2048)
	defer c.Close()

	wb := []byte("UDPCONN TEST")
	if _, err := c.WriteToUDP(wb, c.LocalAddr().(*net.UDPAddr)); err != nil {
		t.Fatalf("net.UDPConn.WriteToUDP failed: %v", err)
	}
	rb := make([]byte, 128)
	if _, _, err := c.ReadFromUDP(rb); err != nil {
		t.Fatalf("net.UDPConn.ReadFromUDP failed: %v", err)
	}
}

func TestIPConnSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Logf("skipping read test on %q", runtime.GOOS)
		return
	}
	if os.Getuid() != 0 {
		t.Logf("skipping test; must be root")
		return
	}

	la, err := net.ResolveIPAddr("ip4", "127.0.0.1")
	if err != nil {
		t.Fatalf("net.ResolveIPAddr failed: %v", err)
	}
	c, err := net.ListenIP("ip4:icmp", la)
	if err != nil {
		t.Fatalf("net.ListenIP failed: %v", err)
	}
	c.File()
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetReadBuffer(2048)
	c.SetWriteBuffer(2048)
	defer c.Close()

	id := os.Getpid() & 0xffff
	wb := newICMPEchoRequest(id, 1, 128, []byte("IPCONN TEST "))
	if _, err := c.WriteToIP(wb, c.LocalAddr().(*net.IPAddr)); err != nil {
		t.Fatalf("net.IPConn.WriteToIP failed: %v", err)
	}
	rb := make([]byte, 20+128)
	if _, _, err := c.ReadFromIP(rb); err != nil {
		t.Fatalf("net.IPConn.ReadFromIP failed: %v", err)
	}
}

// TODO: Find out the use case of ListenUnixgram, I have no idea.
func TestUnixConnSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	}

	p1, p2 := "/tmp/gotest.net1", "/tmp/gotest.net2"
	os.Remove(p1)
	os.Remove(p2)

	a1, err := net.ResolveUnixAddr("unixgram", p1)
	if err != nil {
		t.Fatalf("net.ResolveUnixAddr failed: %v", err)
	}
	c1, err := net.DialUnix("unixgram", a1, nil)
	if err != nil {
		t.Fatalf("net.DialUnix failed: %v", err)
	}
	c1.File()
	c1.LocalAddr()
	c1.RemoteAddr()
	c1.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c1.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c1.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	c1.SetReadBuffer(2048)
	c1.SetWriteBuffer(2048)
	defer c1.Close()
	defer os.Remove(p1)

	a2, err := net.ResolveUnixAddr("unixgram", p2)
	if err != nil {
		t.Fatalf("net.ResolveUnixAddr failed: %v", err)
	}
	c2, err := net.DialUnix("unixgram", a2, nil)
	if err != nil {
		t.Fatalf("net.DialUnix failed: %v", err)
	}
	c2.File()
	c2.LocalAddr()
	c2.RemoteAddr()
	c2.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c2.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c2.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	c2.SetReadBuffer(2048)
	c2.SetWriteBuffer(2048)
	defer c2.Close()
	defer os.Remove(p2)

	wb := []byte("UNIXCONN TEST")
	if _, _, err := c1.WriteMsgUnix(wb, nil, a2); err != nil {
		t.Fatalf("net.UnixConn.WriteMsgUnix failed: %v", err)
	}
	rb2 := make([]byte, 128)
	if _, _, _, _, err := c2.ReadMsgUnix(rb2, nil); err != nil {
		t.Fatalf("net.UnixConn.ReadMsgUnix failed: %v", err)
	}
	if _, err := c2.WriteToUnix(wb, a1); err != nil {
		t.Fatalf("net.UnixConn.WriteToUnix failed: %v", err)
	}
	rb1 := make([]byte, 128)
	if _, _, err := c1.ReadFromUnix(rb1); err != nil {
		t.Fatalf("net.UnixConn.ReadFromUnix failed: %v", err)
	}
}

func newICMPEchoRequest(id, seqnum, msglen int, filler []byte) []byte {
	b := newICMPInfoMessage(id, seqnum, msglen, filler)
	b[0] = 8
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
	b[2] ^= byte(^s & 0xff)
	b[3] ^= byte(^s >> 8)
	return b
}

func newICMPInfoMessage(id, seqnum, msglen int, filler []byte) []byte {
	b := make([]byte, msglen)
	copy(b[8:], bytes.Repeat(filler, (msglen-8)/len(filler)+1))
	b[0] = 0                   // type
	b[1] = 0                   // code
	b[2] = 0                   // checksum
	b[3] = 0                   // checksum
	b[4] = byte(id >> 8)       // identifier
	b[5] = byte(id & 0xff)     // identifier
	b[6] = byte(seqnum >> 8)   // sequence number
	b[7] = byte(seqnum & 0xff) // sequence number
	return b
}
