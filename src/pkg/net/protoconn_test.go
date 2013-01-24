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

var condErrorf = func() func(*testing.T, string, ...interface{}) {
	// A few APIs are not implemented yet on both Plan 9 and Windows.
	switch runtime.GOOS {
	case "plan9", "windows":
		return (*testing.T).Logf
	}
	return (*testing.T).Errorf
}()

func TestTCPListenerSpecificMethods(t *testing.T) {
	la, err := net.ResolveTCPAddr("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.ResolveTCPAddr failed: %v", err)
	}
	ln, err := net.ListenTCP("tcp4", la)
	if err != nil {
		t.Fatalf("net.ListenTCP failed: %v", err)
	}
	ln.Addr()
	ln.SetDeadline(time.Now().Add(30 * time.Nanosecond))
	defer ln.Close()

	if c, err := ln.Accept(); err != nil {
		if !err.(net.Error).Timeout() {
			t.Errorf("net.TCPListener.Accept failed: %v", err)
			return
		}
	} else {
		c.Close()
	}
	if c, err := ln.AcceptTCP(); err != nil {
		if !err.(net.Error).Timeout() {
			t.Errorf("net.TCPListener.AcceptTCP failed: %v", err)
			return
		}
	} else {
		c.Close()
	}

	if f, err := ln.File(); err != nil {
		condErrorf(t, "net.TCPListener.File failed: %v", err)
		return
	} else {
		f.Close()
	}
}

func TestTCPConnSpecificMethods(t *testing.T) {
	la, err := net.ResolveTCPAddr("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.ResolveTCPAddr failed: %v", err)
	}
	ln, err := net.ListenTCP("tcp4", la)
	if err != nil {
		t.Fatalf("net.ListenTCP failed: %v", err)
	}
	ln.Addr()
	defer ln.Close()

	done := make(chan int)
	go transponder(t, ln, done)

	ra, err := net.ResolveTCPAddr("tcp4", ln.Addr().String())
	if err != nil {
		t.Errorf("net.ResolveTCPAddr failed: %v", err)
		return
	}
	c, err := net.DialTCP("tcp4", nil, ra)
	if err != nil {
		t.Errorf("net.DialTCP failed: %v", err)
		return
	}
	c.SetKeepAlive(false)
	c.SetLinger(0)
	c.SetNoDelay(false)
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	defer c.Close()

	if _, err := c.Write([]byte("TCPCONN TEST")); err != nil {
		t.Errorf("net.TCPConn.Write failed: %v", err)
		return
	}
	rb := make([]byte, 128)
	if _, err := c.Read(rb); err != nil {
		t.Errorf("net.TCPConn.Read failed: %v", err)
		return
	}

	<-done
}

func TestUDPConnSpecificMethods(t *testing.T) {
	la, err := net.ResolveUDPAddr("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.ResolveUDPAddr failed: %v", err)
	}
	c, err := net.ListenUDP("udp4", la)
	if err != nil {
		t.Fatalf("net.ListenUDP failed: %v", err)
	}
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	c.SetReadBuffer(2048)
	c.SetWriteBuffer(2048)
	defer c.Close()

	wb := []byte("UDPCONN TEST")
	rb := make([]byte, 128)
	if _, err := c.WriteToUDP(wb, c.LocalAddr().(*net.UDPAddr)); err != nil {
		t.Errorf("net.UDPConn.WriteToUDP failed: %v", err)
		return
	}
	if _, _, err := c.ReadFromUDP(rb); err != nil {
		t.Errorf("net.UDPConn.ReadFromUDP failed: %v", err)
		return
	}
	if _, _, err := c.WriteMsgUDP(wb, nil, c.LocalAddr().(*net.UDPAddr)); err != nil {
		condErrorf(t, "net.UDPConn.WriteMsgUDP failed: %v", err)
		return
	}
	if _, _, _, _, err := c.ReadMsgUDP(rb, nil); err != nil {
		condErrorf(t, "net.UDPConn.ReadMsgUDP failed: %v", err)
		return
	}

	if f, err := c.File(); err != nil {
		condErrorf(t, "net.UDPConn.File failed: %v", err)
		return
	} else {
		f.Close()
	}
}

func TestIPConnSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping read test on %q", runtime.GOOS)
	}
	if os.Getuid() != 0 {
		t.Skipf("skipping test; must be root")
	}

	la, err := net.ResolveIPAddr("ip4", "127.0.0.1")
	if err != nil {
		t.Fatalf("net.ResolveIPAddr failed: %v", err)
	}
	c, err := net.ListenIP("ip4:icmp", la)
	if err != nil {
		t.Fatalf("net.ListenIP failed: %v", err)
	}
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
	rb := make([]byte, 20+128)
	if _, err := c.WriteToIP(wb, c.LocalAddr().(*net.IPAddr)); err != nil {
		t.Errorf("net.IPConn.WriteToIP failed: %v", err)
		return
	}
	if _, _, err := c.ReadFromIP(rb); err != nil {
		t.Errorf("net.IPConn.ReadFromIP failed: %v", err)
		return
	}
	if _, _, err := c.WriteMsgIP(wb, nil, c.LocalAddr().(*net.IPAddr)); err != nil {
		condErrorf(t, "net.UDPConn.WriteMsgIP failed: %v", err)
		return
	}
	if _, _, _, _, err := c.ReadMsgIP(rb, nil); err != nil {
		condErrorf(t, "net.UDPConn.ReadMsgIP failed: %v", err)
		return
	}

	if f, err := c.File(); err != nil {
		condErrorf(t, "net.IPConn.File failed: %v", err)
		return
	} else {
		f.Close()
	}
}

func TestUnixListenerSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping read test on %q", runtime.GOOS)
	}

	p := "/tmp/gotest.net"
	os.Remove(p)
	la, err := net.ResolveUnixAddr("unix", p)
	if err != nil {
		t.Fatalf("net.ResolveUnixAddr failed: %v", err)
	}
	ln, err := net.ListenUnix("unix", la)
	if err != nil {
		t.Fatalf("net.ListenUnix failed: %v", err)
	}
	ln.Addr()
	ln.SetDeadline(time.Now().Add(30 * time.Nanosecond))
	defer ln.Close()
	defer os.Remove(p)

	if c, err := ln.Accept(); err != nil {
		if !err.(net.Error).Timeout() {
			t.Errorf("net.TCPListener.AcceptTCP failed: %v", err)
			return
		}
	} else {
		c.Close()
	}
	if c, err := ln.AcceptUnix(); err != nil {
		if !err.(net.Error).Timeout() {
			t.Errorf("net.TCPListener.AcceptTCP failed: %v", err)
			return
		}
	} else {
		c.Close()
	}

	if f, err := ln.File(); err != nil {
		t.Errorf("net.UnixListener.File failed: %v", err)
		return
	} else {
		f.Close()
	}
}

func TestUnixConnSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	p1, p2, p3 := "/tmp/gotest.net1", "/tmp/gotest.net2", "/tmp/gotest.net3"
	os.Remove(p1)
	os.Remove(p2)
	os.Remove(p3)

	a1, err := net.ResolveUnixAddr("unixgram", p1)
	if err != nil {
		t.Fatalf("net.ResolveUnixAddr failed: %v", err)
	}
	c1, err := net.DialUnix("unixgram", a1, nil)
	if err != nil {
		t.Fatalf("net.DialUnix failed: %v", err)
	}
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
		t.Errorf("net.ResolveUnixAddr failed: %v", err)
		return
	}
	c2, err := net.DialUnix("unixgram", a2, nil)
	if err != nil {
		t.Errorf("net.DialUnix failed: %v", err)
		return
	}
	c2.LocalAddr()
	c2.RemoteAddr()
	c2.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c2.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c2.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	c2.SetReadBuffer(2048)
	c2.SetWriteBuffer(2048)
	defer c2.Close()
	defer os.Remove(p2)

	a3, err := net.ResolveUnixAddr("unixgram", p3)
	if err != nil {
		t.Errorf("net.ResolveUnixAddr failed: %v", err)
		return
	}
	c3, err := net.ListenUnixgram("unixgram", a3)
	if err != nil {
		t.Errorf("net.ListenUnixgram failed: %v", err)
		return
	}
	c3.LocalAddr()
	c3.RemoteAddr()
	c3.SetDeadline(time.Now().Add(100 * time.Millisecond))
	c3.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	c3.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
	c3.SetReadBuffer(2048)
	c3.SetWriteBuffer(2048)
	defer c3.Close()
	defer os.Remove(p3)

	wb := []byte("UNIXCONN TEST")
	rb1 := make([]byte, 128)
	rb2 := make([]byte, 128)
	rb3 := make([]byte, 128)
	if _, _, err := c1.WriteMsgUnix(wb, nil, a2); err != nil {
		t.Errorf("net.UnixConn.WriteMsgUnix failed: %v", err)
		return
	}
	if _, _, _, _, err := c2.ReadMsgUnix(rb2, nil); err != nil {
		t.Errorf("net.UnixConn.ReadMsgUnix failed: %v", err)
		return
	}
	if _, err := c2.WriteToUnix(wb, a1); err != nil {
		t.Errorf("net.UnixConn.WriteToUnix failed: %v", err)
		return
	}
	if _, _, err := c1.ReadFromUnix(rb1); err != nil {
		t.Errorf("net.UnixConn.ReadFromUnix failed: %v", err)
		return
	}
	if _, err := c3.WriteToUnix(wb, a1); err != nil {
		t.Errorf("net.UnixConn.WriteToUnix failed: %v", err)
		return
	}
	if _, _, err := c1.ReadFromUnix(rb1); err != nil {
		t.Errorf("net.UnixConn.ReadFromUnix failed: %v", err)
		return
	}
	if _, err := c2.WriteToUnix(wb, a3); err != nil {
		t.Errorf("net.UnixConn.WriteToUnix failed: %v", err)
		return
	}
	if _, _, err := c3.ReadFromUnix(rb3); err != nil {
		t.Errorf("net.UnixConn.ReadFromUnix failed: %v", err)
		return
	}

	if f, err := c1.File(); err != nil {
		t.Errorf("net.UnixConn.File failed: %v", err)
		return
	} else {
		f.Close()
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
