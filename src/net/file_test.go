// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"reflect"
	"runtime"
	"testing"
)

type listenerFile interface {
	Listener
	File() (f *os.File, err error)
}

type packetConnFile interface {
	PacketConn
	File() (f *os.File, err error)
}

type connFile interface {
	Conn
	File() (f *os.File, err error)
}

func testFileListener(t *testing.T, net, laddr string) {
	switch net {
	case "tcp", "tcp4", "tcp6":
		laddr += ":0" // any available port
	}
	l, err := Listen(net, laddr)
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer l.Close()
	lf := l.(listenerFile)
	f, err := lf.File()
	if err != nil {
		t.Fatalf("File failed: %v", err)
	}
	c, err := FileListener(f)
	if err != nil {
		t.Fatalf("FileListener failed: %v", err)
	}
	if !reflect.DeepEqual(l.Addr(), c.Addr()) {
		t.Fatalf("Addrs not equal: %#v != %#v", l.Addr(), c.Addr())
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

var fileListenerTests = []struct {
	net   string
	laddr string
	ipv6  bool // test with underlying AF_INET6 socket
	linux bool // test with abstract unix domain socket, a Linux-ism
}{
	{net: "tcp", laddr: ""},
	{net: "tcp", laddr: "0.0.0.0"},
	{net: "tcp", laddr: "[::ffff:0.0.0.0]"},
	{net: "tcp", laddr: "[::]", ipv6: true},

	{net: "tcp", laddr: "127.0.0.1"},
	{net: "tcp", laddr: "[::ffff:127.0.0.1]"},
	{net: "tcp", laddr: "[::1]", ipv6: true},

	{net: "tcp4", laddr: ""},
	{net: "tcp4", laddr: "0.0.0.0"},
	{net: "tcp4", laddr: "[::ffff:0.0.0.0]"},

	{net: "tcp4", laddr: "127.0.0.1"},
	{net: "tcp4", laddr: "[::ffff:127.0.0.1]"},

	{net: "tcp6", laddr: "", ipv6: true},
	{net: "tcp6", laddr: "[::]", ipv6: true},

	{net: "tcp6", laddr: "[::1]", ipv6: true},

	{net: "unix", laddr: "@gotest/net", linux: true},
	{net: "unixpacket", laddr: "@gotest/net", linux: true},
}

func TestFileListener(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "windows":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	for _, tt := range fileListenerTests {
		if skipServerTest(tt.net, "unix", tt.laddr, tt.ipv6, false, tt.linux) {
			continue
		}
		if skipServerTest(tt.net, "unixpacket", tt.laddr, tt.ipv6, false, tt.linux) {
			continue
		}
		testFileListener(t, tt.net, tt.laddr)
	}
}

func testFilePacketConn(t *testing.T, pcf packetConnFile, listen bool) {
	f, err := pcf.File()
	if err != nil {
		t.Fatalf("File failed: %v", err)
	}
	c, err := FilePacketConn(f)
	if err != nil {
		t.Fatalf("FilePacketConn failed: %v", err)
	}
	if !reflect.DeepEqual(pcf.LocalAddr(), c.LocalAddr()) {
		t.Fatalf("LocalAddrs not equal: %#v != %#v", pcf.LocalAddr(), c.LocalAddr())
	}
	if listen {
		if _, err := c.WriteTo([]byte{}, c.LocalAddr()); err != nil {
			t.Fatalf("WriteTo failed: %v", err)
		}
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

func testFilePacketConnListen(t *testing.T, net, laddr string) {
	switch net {
	case "udp", "udp4", "udp6":
		laddr += ":0" // any available port
	}
	l, err := ListenPacket(net, laddr)
	if err != nil {
		t.Fatalf("ListenPacket failed: %v", err)
	}
	testFilePacketConn(t, l.(packetConnFile), true)
	if err := l.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

func testFilePacketConnDial(t *testing.T, net, raddr string) {
	switch net {
	case "udp", "udp4", "udp6":
		raddr += ":12345"
	}
	c, err := Dial(net, raddr)
	if err != nil {
		t.Fatalf("Dial failed: %v", err)
	}
	testFilePacketConn(t, c.(packetConnFile), false)
	if err := c.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

var filePacketConnTests = []struct {
	net   string
	addr  string
	ipv6  bool // test with underlying AF_INET6 socket
	linux bool // test with abstract unix domain socket, a Linux-ism
}{
	{net: "udp", addr: "127.0.0.1"},
	{net: "udp", addr: "[::ffff:127.0.0.1]"},
	{net: "udp", addr: "[::1]", ipv6: true},

	{net: "udp4", addr: "127.0.0.1"},
	{net: "udp4", addr: "[::ffff:127.0.0.1]"},

	{net: "udp6", addr: "[::1]", ipv6: true},

	{net: "ip4:icmp", addr: "127.0.0.1"},

	{net: "unixgram", addr: "@gotest3/net", linux: true},
}

func TestFilePacketConn(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "windows":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	for _, tt := range filePacketConnTests {
		if skipServerTest(tt.net, "unixgram", tt.addr, tt.ipv6, false, tt.linux) {
			continue
		}
		if os.Getuid() != 0 && tt.net == "ip4:icmp" {
			t.Log("skipping test; must be root")
			continue
		}
		testFilePacketConnListen(t, tt.net, tt.addr)
		switch tt.addr {
		case "", "0.0.0.0", "[::ffff:0.0.0.0]", "[::]":
		default:
			if tt.net != "unixgram" {
				testFilePacketConnDial(t, tt.net, tt.addr)
			}
		}
	}
}
