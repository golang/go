// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements API tests across platforms and will never have a build
// tag.

package net

import (
	"io/ioutil"
	"os"
	"runtime"
	"testing"
	"time"
)

// testUnixAddr uses ioutil.TempFile to get a name that is unique. It
// also uses /tmp directory in case it is prohibited to create UNIX
// sockets in TMPDIR.
func testUnixAddr() string {
	f, err := ioutil.TempFile("/tmp", "nettest")
	if err != nil {
		panic(err)
	}
	addr := f.Name()
	f.Close()
	os.Remove(addr)
	return addr
}

var condFatalf = func() func(*testing.T, string, ...interface{}) {
	// A few APIs are not implemented yet on both Plan 9 and Windows.
	switch runtime.GOOS {
	case "plan9", "windows":
		return (*testing.T).Logf
	}
	return (*testing.T).Fatalf
}()

func TestTCPListenerSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	la, err := ResolveTCPAddr("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("ResolveTCPAddr failed: %v", err)
	}
	ln, err := ListenTCP("tcp4", la)
	if err != nil {
		t.Fatalf("ListenTCP failed: %v", err)
	}
	defer ln.Close()
	ln.Addr()
	ln.SetDeadline(time.Now().Add(30 * time.Nanosecond))

	if c, err := ln.Accept(); err != nil {
		if !err.(Error).Timeout() {
			t.Fatalf("TCPListener.Accept failed: %v", err)
		}
	} else {
		c.Close()
	}
	if c, err := ln.AcceptTCP(); err != nil {
		if !err.(Error).Timeout() {
			t.Fatalf("TCPListener.AcceptTCP failed: %v", err)
		}
	} else {
		c.Close()
	}

	if f, err := ln.File(); err != nil {
		condFatalf(t, "TCPListener.File failed: %v", err)
	} else {
		f.Close()
	}
}

func TestTCPConnSpecificMethods(t *testing.T) {
	la, err := ResolveTCPAddr("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("ResolveTCPAddr failed: %v", err)
	}
	ln, err := ListenTCP("tcp4", la)
	if err != nil {
		t.Fatalf("ListenTCP failed: %v", err)
	}
	defer ln.Close()
	ln.Addr()

	done := make(chan int)
	go transponder(t, ln, done)

	ra, err := ResolveTCPAddr("tcp4", ln.Addr().String())
	if err != nil {
		t.Fatalf("ResolveTCPAddr failed: %v", err)
	}
	c, err := DialTCP("tcp4", nil, ra)
	if err != nil {
		t.Fatalf("DialTCP failed: %v", err)
	}
	defer c.Close()
	c.SetKeepAlive(false)
	c.SetKeepAlivePeriod(3 * time.Second)
	c.SetLinger(0)
	c.SetNoDelay(false)
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))

	if _, err := c.Write([]byte("TCPCONN TEST")); err != nil {
		t.Fatalf("TCPConn.Write failed: %v", err)
	}
	rb := make([]byte, 128)
	if _, err := c.Read(rb); err != nil {
		t.Fatalf("TCPConn.Read failed: %v", err)
	}

	<-done
}

func TestUDPConnSpecificMethods(t *testing.T) {
	la, err := ResolveUDPAddr("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("ResolveUDPAddr failed: %v", err)
	}
	c, err := ListenUDP("udp4", la)
	if err != nil {
		t.Fatalf("ListenUDP failed: %v", err)
	}
	defer c.Close()
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))
	c.SetReadBuffer(2048)
	c.SetWriteBuffer(2048)

	wb := []byte("UDPCONN TEST")
	rb := make([]byte, 128)
	if _, err := c.WriteToUDP(wb, c.LocalAddr().(*UDPAddr)); err != nil {
		t.Fatalf("UDPConn.WriteToUDP failed: %v", err)
	}
	if _, _, err := c.ReadFromUDP(rb); err != nil {
		t.Fatalf("UDPConn.ReadFromUDP failed: %v", err)
	}
	if _, _, err := c.WriteMsgUDP(wb, nil, c.LocalAddr().(*UDPAddr)); err != nil {
		condFatalf(t, "UDPConn.WriteMsgUDP failed: %v", err)
	}
	if _, _, _, _, err := c.ReadMsgUDP(rb, nil); err != nil {
		condFatalf(t, "UDPConn.ReadMsgUDP failed: %v", err)
	}

	if f, err := c.File(); err != nil {
		condFatalf(t, "UDPConn.File failed: %v", err)
	} else {
		f.Close()
	}

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("UDPConn.WriteToUDP or WriteMsgUDP panicked: %v", p)
		}
	}()

	c.WriteToUDP(wb, nil)
	c.WriteMsgUDP(wb, nil, nil)
}

func TestIPConnSpecificMethods(t *testing.T) {
	if skip, skipmsg := skipRawSocketTest(t); skip {
		t.Skip(skipmsg)
	}

	la, err := ResolveIPAddr("ip4", "127.0.0.1")
	if err != nil {
		t.Fatalf("ResolveIPAddr failed: %v", err)
	}
	c, err := ListenIP("ip4:icmp", la)
	if err != nil {
		t.Fatalf("ListenIP failed: %v", err)
	}
	defer c.Close()
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))
	c.SetReadBuffer(2048)
	c.SetWriteBuffer(2048)

	wb, err := (&icmpMessage{
		Type: icmpv4EchoRequest, Code: 0,
		Body: &icmpEcho{
			ID: os.Getpid() & 0xffff, Seq: 1,
			Data: []byte("IPCONN TEST "),
		},
	}).Marshal()
	if err != nil {
		t.Fatalf("icmpMessage.Marshal failed: %v", err)
	}
	rb := make([]byte, 20+len(wb))
	if _, err := c.WriteToIP(wb, c.LocalAddr().(*IPAddr)); err != nil {
		t.Fatalf("IPConn.WriteToIP failed: %v", err)
	}
	if _, _, err := c.ReadFromIP(rb); err != nil {
		t.Fatalf("IPConn.ReadFromIP failed: %v", err)
	}
	if _, _, err := c.WriteMsgIP(wb, nil, c.LocalAddr().(*IPAddr)); err != nil {
		condFatalf(t, "IPConn.WriteMsgIP failed: %v", err)
	}
	if _, _, _, _, err := c.ReadMsgIP(rb, nil); err != nil {
		condFatalf(t, "IPConn.ReadMsgIP failed: %v", err)
	}

	if f, err := c.File(); err != nil {
		condFatalf(t, "IPConn.File failed: %v", err)
	} else {
		f.Close()
	}

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("IPConn.WriteToIP or WriteMsgIP panicked: %v", p)
		}
	}()

	c.WriteToIP(wb, nil)
	c.WriteMsgIP(wb, nil, nil)
}

func TestUnixListenerSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	addr := testUnixAddr()
	la, err := ResolveUnixAddr("unix", addr)
	if err != nil {
		t.Fatalf("ResolveUnixAddr failed: %v", err)
	}
	ln, err := ListenUnix("unix", la)
	if err != nil {
		t.Fatalf("ListenUnix failed: %v", err)
	}
	defer ln.Close()
	defer os.Remove(addr)
	ln.Addr()
	ln.SetDeadline(time.Now().Add(30 * time.Nanosecond))

	if c, err := ln.Accept(); err != nil {
		if !err.(Error).Timeout() {
			t.Fatalf("UnixListener.Accept failed: %v", err)
		}
	} else {
		c.Close()
	}
	if c, err := ln.AcceptUnix(); err != nil {
		if !err.(Error).Timeout() {
			t.Fatalf("UnixListener.AcceptUnix failed: %v", err)
		}
	} else {
		c.Close()
	}

	if f, err := ln.File(); err != nil {
		t.Fatalf("UnixListener.File failed: %v", err)
	} else {
		f.Close()
	}
}

func TestUnixConnSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	addr1, addr2, addr3 := testUnixAddr(), testUnixAddr(), testUnixAddr()

	a1, err := ResolveUnixAddr("unixgram", addr1)
	if err != nil {
		t.Fatalf("ResolveUnixAddr failed: %v", err)
	}
	c1, err := DialUnix("unixgram", a1, nil)
	if err != nil {
		t.Fatalf("DialUnix failed: %v", err)
	}
	defer c1.Close()
	defer os.Remove(addr1)
	c1.LocalAddr()
	c1.RemoteAddr()
	c1.SetDeadline(time.Now().Add(someTimeout))
	c1.SetReadDeadline(time.Now().Add(someTimeout))
	c1.SetWriteDeadline(time.Now().Add(someTimeout))
	c1.SetReadBuffer(2048)
	c1.SetWriteBuffer(2048)

	a2, err := ResolveUnixAddr("unixgram", addr2)
	if err != nil {
		t.Fatalf("ResolveUnixAddr failed: %v", err)
	}
	c2, err := DialUnix("unixgram", a2, nil)
	if err != nil {
		t.Fatalf("DialUnix failed: %v", err)
	}
	defer c2.Close()
	defer os.Remove(addr2)
	c2.LocalAddr()
	c2.RemoteAddr()
	c2.SetDeadline(time.Now().Add(someTimeout))
	c2.SetReadDeadline(time.Now().Add(someTimeout))
	c2.SetWriteDeadline(time.Now().Add(someTimeout))
	c2.SetReadBuffer(2048)
	c2.SetWriteBuffer(2048)

	a3, err := ResolveUnixAddr("unixgram", addr3)
	if err != nil {
		t.Fatalf("ResolveUnixAddr failed: %v", err)
	}
	c3, err := ListenUnixgram("unixgram", a3)
	if err != nil {
		t.Fatalf("ListenUnixgram failed: %v", err)
	}
	defer c3.Close()
	defer os.Remove(addr3)
	c3.LocalAddr()
	c3.RemoteAddr()
	c3.SetDeadline(time.Now().Add(someTimeout))
	c3.SetReadDeadline(time.Now().Add(someTimeout))
	c3.SetWriteDeadline(time.Now().Add(someTimeout))
	c3.SetReadBuffer(2048)
	c3.SetWriteBuffer(2048)

	wb := []byte("UNIXCONN TEST")
	rb1 := make([]byte, 128)
	rb2 := make([]byte, 128)
	rb3 := make([]byte, 128)
	if _, _, err := c1.WriteMsgUnix(wb, nil, a2); err != nil {
		t.Fatalf("UnixConn.WriteMsgUnix failed: %v", err)
	}
	if _, _, _, _, err := c2.ReadMsgUnix(rb2, nil); err != nil {
		t.Fatalf("UnixConn.ReadMsgUnix failed: %v", err)
	}
	if _, err := c2.WriteToUnix(wb, a1); err != nil {
		t.Fatalf("UnixConn.WriteToUnix failed: %v", err)
	}
	if _, _, err := c1.ReadFromUnix(rb1); err != nil {
		t.Fatalf("UnixConn.ReadFromUnix failed: %v", err)
	}
	if _, err := c3.WriteToUnix(wb, a1); err != nil {
		t.Fatalf("UnixConn.WriteToUnix failed: %v", err)
	}
	if _, _, err := c1.ReadFromUnix(rb1); err != nil {
		t.Fatalf("UnixConn.ReadFromUnix failed: %v", err)
	}
	if _, err := c2.WriteToUnix(wb, a3); err != nil {
		t.Fatalf("UnixConn.WriteToUnix failed: %v", err)
	}
	if _, _, err := c3.ReadFromUnix(rb3); err != nil {
		t.Fatalf("UnixConn.ReadFromUnix failed: %v", err)
	}

	if f, err := c1.File(); err != nil {
		t.Fatalf("UnixConn.File failed: %v", err)
	} else {
		f.Close()
	}

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("UnixConn.WriteToUnix or WriteMsgUnix panicked: %v", p)
		}
	}()

	c1.WriteToUnix(wb, nil)
	c1.WriteMsgUnix(wb, nil, nil)
	c3.WriteToUnix(wb, nil)
	c3.WriteMsgUnix(wb, nil, nil)
}
