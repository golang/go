// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements API tests across platforms and will never have a build
// tag.

//go:build !js
// +build !js

package net

import (
	"os"
	"runtime"
	"testing"
	"time"
)

// The full stack test cases for IPConn have been moved to the
// following:
//	golang.org/x/net/ipv4
//	golang.org/x/net/ipv6
//	golang.org/x/net/icmp

func TestTCPListenerSpecificMethods(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	la, err := ResolveTCPAddr("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ln, err := ListenTCP("tcp4", la)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	ln.Addr()
	ln.SetDeadline(time.Now().Add(30 * time.Nanosecond))

	if c, err := ln.Accept(); err != nil {
		if !err.(Error).Timeout() {
			t.Fatal(err)
		}
	} else {
		c.Close()
	}
	if c, err := ln.AcceptTCP(); err != nil {
		if !err.(Error).Timeout() {
			t.Fatal(err)
		}
	} else {
		c.Close()
	}

	if f, err := ln.File(); err != nil {
		condFatalf(t, "file+net", "%v", err)
	} else {
		f.Close()
	}
}

func TestTCPConnSpecificMethods(t *testing.T) {
	la, err := ResolveTCPAddr("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	ln, err := ListenTCP("tcp4", la)
	if err != nil {
		t.Fatal(err)
	}
	ch := make(chan error, 1)
	handler := func(ls *localServer, ln Listener) { ls.transponder(ls.Listener, ch) }
	ls, err := (&streamListener{Listener: ln}).newLocalServer()
	if err != nil {
		t.Fatal(err)
	}
	defer ls.teardown()
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}

	ra, err := ResolveTCPAddr("tcp4", ls.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	c, err := DialTCP("tcp4", nil, ra)
	if err != nil {
		t.Fatal(err)
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
		t.Fatal(err)
	}
	rb := make([]byte, 128)
	if _, err := c.Read(rb); err != nil {
		t.Fatal(err)
	}

	for err := range ch {
		t.Error(err)
	}
}

func TestUDPConnSpecificMethods(t *testing.T) {
	la, err := ResolveUDPAddr("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenUDP("udp4", la)
	if err != nil {
		t.Fatal(err)
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
		t.Fatal(err)
	}
	if _, _, err := c.ReadFromUDP(rb); err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.WriteMsgUDP(wb, nil, c.LocalAddr().(*UDPAddr)); err != nil {
		condFatalf(t, c.LocalAddr().Network(), "%v", err)
	}
	if _, _, _, _, err := c.ReadMsgUDP(rb, nil); err != nil {
		condFatalf(t, c.LocalAddr().Network(), "%v", err)
	}

	if f, err := c.File(); err != nil {
		condFatalf(t, "file+net", "%v", err)
	} else {
		f.Close()
	}

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("panicked: %v", p)
		}
	}()

	c.WriteToUDP(wb, nil)
	c.WriteMsgUDP(wb, nil, nil)
}

func TestIPConnSpecificMethods(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("must be root")
	}

	la, err := ResolveIPAddr("ip4", "127.0.0.1")
	if err != nil {
		t.Fatal(err)
	}
	c, err := ListenIP("ip4:icmp", la)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.LocalAddr()
	c.RemoteAddr()
	c.SetDeadline(time.Now().Add(someTimeout))
	c.SetReadDeadline(time.Now().Add(someTimeout))
	c.SetWriteDeadline(time.Now().Add(someTimeout))
	c.SetReadBuffer(2048)
	c.SetWriteBuffer(2048)

	if f, err := c.File(); err != nil {
		condFatalf(t, "file+net", "%v", err)
	} else {
		f.Close()
	}

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("panicked: %v", p)
		}
	}()

	wb := []byte("IPCONN TEST")
	c.WriteToIP(wb, nil)
	c.WriteMsgIP(wb, nil, nil)
}

func TestUnixListenerSpecificMethods(t *testing.T) {
	if !testableNetwork("unix") {
		t.Skip("unix test")
	}

	addr := testUnixAddr()
	la, err := ResolveUnixAddr("unix", addr)
	if err != nil {
		t.Fatal(err)
	}
	ln, err := ListenUnix("unix", la)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	defer os.Remove(addr)
	ln.Addr()
	ln.SetDeadline(time.Now().Add(30 * time.Nanosecond))

	if c, err := ln.Accept(); err != nil {
		if !err.(Error).Timeout() {
			t.Fatal(err)
		}
	} else {
		c.Close()
	}
	if c, err := ln.AcceptUnix(); err != nil {
		if !err.(Error).Timeout() {
			t.Fatal(err)
		}
	} else {
		c.Close()
	}

	if f, err := ln.File(); err != nil {
		t.Fatal(err)
	} else {
		f.Close()
	}
}

func TestUnixConnSpecificMethods(t *testing.T) {
	if !testableNetwork("unixgram") {
		t.Skip("unixgram test")
	}

	addr1, addr2, addr3 := testUnixAddr(), testUnixAddr(), testUnixAddr()

	a1, err := ResolveUnixAddr("unixgram", addr1)
	if err != nil {
		t.Fatal(err)
	}
	c1, err := DialUnix("unixgram", a1, nil)
	if err != nil {
		t.Fatal(err)
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
		t.Fatal(err)
	}
	c2, err := DialUnix("unixgram", a2, nil)
	if err != nil {
		t.Fatal(err)
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
		t.Fatal(err)
	}
	c3, err := ListenUnixgram("unixgram", a3)
	if err != nil {
		t.Fatal(err)
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
		t.Fatal(err)
	}
	if _, _, _, _, err := c2.ReadMsgUnix(rb2, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := c2.WriteToUnix(wb, a1); err != nil {
		t.Fatal(err)
	}
	if _, _, err := c1.ReadFromUnix(rb1); err != nil {
		t.Fatal(err)
	}
	if _, err := c3.WriteToUnix(wb, a1); err != nil {
		t.Fatal(err)
	}
	if _, _, err := c1.ReadFromUnix(rb1); err != nil {
		t.Fatal(err)
	}
	if _, err := c2.WriteToUnix(wb, a3); err != nil {
		t.Fatal(err)
	}
	if _, _, err := c3.ReadFromUnix(rb3); err != nil {
		t.Fatal(err)
	}

	if f, err := c1.File(); err != nil {
		t.Fatal(err)
	} else {
		f.Close()
	}

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("panicked: %v", p)
		}
	}()

	c1.WriteToUnix(wb, nil)
	c1.WriteMsgUnix(wb, nil, nil)
	c3.WriteToUnix(wb, nil)
	c3.WriteMsgUnix(wb, nil, nil)
}
