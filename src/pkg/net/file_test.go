// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"reflect"
	"runtime"
	"syscall"
	"testing"
)

type listenerFile interface {
	Listener
	File() (f *os.File, err os.Error)
}

type packetConnFile interface {
	PacketConn
	File() (f *os.File, err os.Error)
}

type connFile interface {
	Conn
	File() (f *os.File, err os.Error)
}

func testFileListener(t *testing.T, net, laddr string) {
	if net == "tcp" {
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

func TestFileListener(t *testing.T) {
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		return
	}
	testFileListener(t, "tcp", "127.0.0.1")
	testFileListener(t, "tcp", "127.0.0.1")
	if supportsIPv6 && supportsIPv4map {
		testFileListener(t, "tcp", "[::ffff:127.0.0.1]")
		testFileListener(t, "tcp", "127.0.0.1")
		testFileListener(t, "tcp", "[::ffff:127.0.0.1]")
	}
	if syscall.OS == "linux" {
		testFileListener(t, "unix", "@gotest/net")
		testFileListener(t, "unixpacket", "@gotest/net")
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
	l, err := ListenPacket(net, laddr)
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	testFilePacketConn(t, l.(packetConnFile), true)
	if err := l.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

func testFilePacketConnDial(t *testing.T, net, raddr string) {
	c, err := Dial(net, raddr)
	if err != nil {
		t.Fatalf("Dial failed: %v", err)
	}
	testFilePacketConn(t, c.(packetConnFile), false)
	if err := c.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
}

func TestFilePacketConn(t *testing.T) {
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		return
	}
	testFilePacketConnListen(t, "udp", "127.0.0.1:0")
	testFilePacketConnDial(t, "udp", "127.0.0.1:12345")
	if supportsIPv6 {
		testFilePacketConnListen(t, "udp", "[::1]:0")
	}
	if supportsIPv6 && supportsIPv4map {
		testFilePacketConnDial(t, "udp", "[::ffff:127.0.0.1]:12345")
	}
	if syscall.OS == "linux" {
		testFilePacketConnListen(t, "unixgram", "@gotest1/net")
	}
}
