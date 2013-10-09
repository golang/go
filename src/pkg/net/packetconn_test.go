// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements API tests across platforms and will never have a build
// tag.

package net

import (
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

func strfunc(s string) func() string {
	return func() string {
		return s
	}
}

func packetConnTestData(t *testing.T, net string, i int) ([]byte, func()) {
	switch net {
	case "udp":
		return []byte("UDP PACKETCONN TEST"), nil
	case "ip":
		if skip, skipmsg := skipRawSocketTest(t); skip {
			return nil, func() {
				t.Logf(skipmsg)
			}
		}
		b, err := (&icmpMessage{
			Type: icmpv4EchoRequest, Code: 0,
			Body: &icmpEcho{
				ID: os.Getpid() & 0xffff, Seq: i + 1,
				Data: []byte("IP PACKETCONN TEST"),
			},
		}).Marshal()
		if err != nil {
			return nil, func() {
				t.Fatalf("icmpMessage.Marshal failed: %v", err)
			}
		}
		return b, nil
	case "unixgram":
		switch runtime.GOOS {
		case "plan9", "windows":
			return nil, func() {
				t.Logf("skipping %q test on %q", net, runtime.GOOS)
			}
		default:
			return []byte("UNIXGRAM PACKETCONN TEST"), nil
		}
	default:
		return nil, func() {
			t.Logf("skipping %q test", net)
		}
	}
}

var packetConnTests = []struct {
	net   string
	addr1 func() string
	addr2 func() string
}{
	{"udp", strfunc("127.0.0.1:0"), strfunc("127.0.0.1:0")},
	{"ip:icmp", strfunc("127.0.0.1"), strfunc("127.0.0.1")},
	{"unixgram", testUnixAddr, testUnixAddr},
}

func TestPacketConn(t *testing.T) {
	closer := func(c PacketConn, net, addr1, addr2 string) {
		c.Close()
		switch net {
		case "unixgram":
			os.Remove(addr1)
			os.Remove(addr2)
		}
	}

	for i, tt := range packetConnTests {
		netstr := strings.Split(tt.net, ":")
		wb, skipOrFatalFn := packetConnTestData(t, netstr[0], i)
		if skipOrFatalFn != nil {
			skipOrFatalFn()
			continue
		}

		addr1, addr2 := tt.addr1(), tt.addr2()
		c1, err := ListenPacket(tt.net, addr1)
		if err != nil {
			t.Fatalf("ListenPacket failed: %v", err)
		}
		defer closer(c1, netstr[0], addr1, addr2)
		c1.LocalAddr()
		c1.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c1.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c1.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))

		c2, err := ListenPacket(tt.net, addr2)
		if err != nil {
			t.Fatalf("ListenPacket failed: %v", err)
		}
		defer closer(c2, netstr[0], addr1, addr2)
		c2.LocalAddr()
		c2.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c2.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c2.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))

		if _, err := c1.WriteTo(wb, c2.LocalAddr()); err != nil {
			t.Fatalf("PacketConn.WriteTo failed: %v", err)
		}
		rb2 := make([]byte, 128)
		if _, _, err := c2.ReadFrom(rb2); err != nil {
			t.Fatalf("PacketConn.ReadFrom failed: %v", err)
		}
		if _, err := c2.WriteTo(wb, c1.LocalAddr()); err != nil {
			t.Fatalf("PacketConn.WriteTo failed: %v", err)
		}
		rb1 := make([]byte, 128)
		if _, _, err := c1.ReadFrom(rb1); err != nil {
			t.Fatalf("PacketConn.ReadFrom failed: %v", err)
		}
	}
}

func TestConnAndPacketConn(t *testing.T) {
	closer := func(c PacketConn, net, addr1, addr2 string) {
		c.Close()
		switch net {
		case "unixgram":
			os.Remove(addr1)
			os.Remove(addr2)
		}
	}

	for i, tt := range packetConnTests {
		var wb []byte
		netstr := strings.Split(tt.net, ":")
		wb, skipOrFatalFn := packetConnTestData(t, netstr[0], i)
		if skipOrFatalFn != nil {
			skipOrFatalFn()
			continue
		}

		addr1, addr2 := tt.addr1(), tt.addr2()
		c1, err := ListenPacket(tt.net, addr1)
		if err != nil {
			t.Fatalf("ListenPacket failed: %v", err)
		}
		defer closer(c1, netstr[0], addr1, addr2)
		c1.LocalAddr()
		c1.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c1.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c1.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))

		c2, err := Dial(tt.net, c1.LocalAddr().String())
		if err != nil {
			t.Fatalf("Dial failed: %v", err)
		}
		defer c2.Close()
		c2.LocalAddr()
		c2.RemoteAddr()
		c2.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c2.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c2.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))

		if _, err := c2.Write(wb); err != nil {
			t.Fatalf("Conn.Write failed: %v", err)
		}
		rb1 := make([]byte, 128)
		if _, _, err := c1.ReadFrom(rb1); err != nil {
			t.Fatalf("PacketConn.ReadFrom failed: %v", err)
		}
		var dst Addr
		switch netstr[0] {
		case "ip":
			dst = &IPAddr{IP: IPv4(127, 0, 0, 1)}
		case "unixgram":
			continue
		default:
			dst = c2.LocalAddr()
		}
		if _, err := c1.WriteTo(wb, dst); err != nil {
			t.Fatalf("PacketConn.WriteTo failed: %v", err)
		}
		rb2 := make([]byte, 128)
		if _, err := c2.Read(rb2); err != nil {
			t.Fatalf("Conn.Read failed: %v", err)
		}
	}
}
