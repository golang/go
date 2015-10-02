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
	l, err := Listen(net, laddr)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	lf := l.(listenerFile)
	f, err := lf.File()
	if err != nil {
		t.Fatal(err)
	}
	c, err := FileListener(f)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(l.Addr(), c.Addr()) {
		t.Fatalf("got %#v; want%#v", l.Addr(), c.Addr())
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

var fileListenerTests = []struct {
	net   string
	laddr string
}{
	{net: "tcp", laddr: ":0"},
	{net: "tcp", laddr: "0.0.0.0:0"},
	{net: "tcp", laddr: "[::ffff:0.0.0.0]:0"},
	{net: "tcp", laddr: "[::]:0"},

	{net: "tcp", laddr: "127.0.0.1:0"},
	{net: "tcp", laddr: "[::ffff:127.0.0.1]:0"},
	{net: "tcp", laddr: "[::1]:0"},

	{net: "tcp4", laddr: ":0"},
	{net: "tcp4", laddr: "0.0.0.0:0"},
	{net: "tcp4", laddr: "[::ffff:0.0.0.0]:0"},

	{net: "tcp4", laddr: "127.0.0.1:0"},
	{net: "tcp4", laddr: "[::ffff:127.0.0.1]:0"},

	{net: "tcp6", laddr: ":0"},
	{net: "tcp6", laddr: "[::]:0"},

	{net: "tcp6", laddr: "[::1]:0"},

	{net: "unix", laddr: "@gotest/net"},
	{net: "unixpacket", laddr: "@gotest/net"},
}

func TestFileListener(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, tt := range fileListenerTests {
		if !testableListenArgs(tt.net, tt.laddr, "") {
			t.Logf("skipping %s test", tt.net+" "+tt.laddr)
			continue
		}
		testFileListener(t, tt.net, tt.laddr)
	}
}

func testFilePacketConn(t *testing.T, pcf packetConnFile, listen bool) {
	f, err := pcf.File()
	if err != nil {
		t.Fatal(err)
	}
	c, err := FilePacketConn(f)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pcf.LocalAddr(), c.LocalAddr()) {
		t.Fatalf("got %#v; want %#v", pcf.LocalAddr(), c.LocalAddr())
	}
	if listen {
		if _, err := c.WriteTo([]byte{}, c.LocalAddr()); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func testFilePacketConnListen(t *testing.T, net, laddr string) {
	l, err := ListenPacket(net, laddr)
	if err != nil {
		t.Fatal(err)
	}
	testFilePacketConn(t, l.(packetConnFile), true)
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
}

func testFilePacketConnDial(t *testing.T, net, raddr string) {
	c, err := Dial(net, raddr)
	if err != nil {
		t.Fatal(err)
	}
	testFilePacketConn(t, c.(packetConnFile), false)
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
}

var filePacketConnTests = []struct {
	net  string
	addr string
}{
	{net: "udp", addr: "127.0.0.1:0"},
	{net: "udp", addr: "[::ffff:127.0.0.1]:0"},
	{net: "udp", addr: "[::1]:0"},

	{net: "udp4", addr: "127.0.0.1:0"},
	{net: "udp4", addr: "[::ffff:127.0.0.1]:0"},

	{net: "udp6", addr: "[::1]:0"},

	// TODO(mikioh,bradfitz): reenable once 10730 is fixed
	// {net: "ip4:icmp", addr: "127.0.0.1"},

	{net: "unixgram", addr: "@gotest3/net"},
}

func TestFilePacketConn(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9", "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, tt := range filePacketConnTests {
		if !testableListenArgs(tt.net, tt.addr, "") {
			t.Logf("skipping %s test", tt.net+" "+tt.addr)
			continue
		}
		if os.Getuid() != 0 && tt.net == "ip4:icmp" {
			t.Log("skipping test; must be root")
			continue
		}
		testFilePacketConnListen(t, tt.net, tt.addr)
		switch tt.net {
		case "udp", "udp4", "udp6":
			host, _, err := SplitHostPort(tt.addr)
			if err != nil {
				t.Error(err)
				continue
			}
			testFilePacketConnDial(t, tt.net, JoinHostPort(host, "12345"))
		case "ip4:icmp":
			testFilePacketConnDial(t, tt.net, tt.addr)
		}
	}
}
