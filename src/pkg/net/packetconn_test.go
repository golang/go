// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net_test

import (
	"net"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

var packetConnTests = []struct {
	net   string
	addr1 string
	addr2 string
}{
	{"udp", "127.0.0.1:0", "127.0.0.1:0"},
	{"ip:icmp", "127.0.0.1", "127.0.0.1"},
	{"unixgram", "/tmp/gotest.net1", "/tmp/gotest.net2"},
}

func TestPacketConn(t *testing.T) {
	closer := func(c net.PacketConn, net, addr1, addr2 string) {
		c.Close()
		switch net {
		case "unixgram":
			os.Remove(addr1)
			os.Remove(addr2)
		}
	}

	for _, tt := range packetConnTests {
		var wb []byte
		netstr := strings.Split(tt.net, ":")
		switch netstr[0] {
		case "udp":
			wb = []byte("UDP PACKETCONN TEST")
		case "ip":
			switch runtime.GOOS {
			case "plan9":
				continue
			}
			if os.Getuid() != 0 {
				continue
			}
			id := os.Getpid() & 0xffff
			wb = newICMPEchoRequest(id, 1, 128, []byte("IP PACKETCONN TEST"))
		case "unixgram":
			switch runtime.GOOS {
			case "plan9", "windows":
				continue
			}
			os.Remove(tt.addr1)
			os.Remove(tt.addr2)
			wb = []byte("UNIXGRAM PACKETCONN TEST")
		default:
			continue
		}

		c1, err := net.ListenPacket(tt.net, tt.addr1)
		if err != nil {
			t.Fatalf("net.ListenPacket failed: %v", err)
		}
		c1.LocalAddr()
		c1.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c1.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c1.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
		defer closer(c1, netstr[0], tt.addr1, tt.addr2)

		c2, err := net.ListenPacket(tt.net, tt.addr2)
		if err != nil {
			t.Fatalf("net.ListenPacket failed: %v", err)
		}
		c2.LocalAddr()
		c2.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c2.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c2.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
		defer closer(c2, netstr[0], tt.addr1, tt.addr2)

		if _, err := c1.WriteTo(wb, c2.LocalAddr()); err != nil {
			t.Fatalf("net.PacketConn.WriteTo failed: %v", err)
		}
		rb2 := make([]byte, 128)
		if _, _, err := c2.ReadFrom(rb2); err != nil {
			t.Fatalf("net.PacketConn.ReadFrom failed: %v", err)
		}
		if _, err := c2.WriteTo(wb, c1.LocalAddr()); err != nil {
			t.Fatalf("net.PacketConn.WriteTo failed: %v", err)
		}
		rb1 := make([]byte, 128)
		if _, _, err := c1.ReadFrom(rb1); err != nil {
			t.Fatalf("net.PacketConn.ReadFrom failed: %v", err)
		}
	}
}

func TestConnAndPacketConn(t *testing.T) {
	for _, tt := range packetConnTests {
		var wb []byte
		netstr := strings.Split(tt.net, ":")
		switch netstr[0] {
		case "udp":
			wb = []byte("UDP PACKETCONN TEST")
		case "ip":
			switch runtime.GOOS {
			case "plan9":
				continue
			}
			if os.Getuid() != 0 {
				continue
			}
			id := os.Getpid() & 0xffff
			wb = newICMPEchoRequest(id, 1, 128, []byte("IP PACKETCONN TEST"))
		default:
			continue
		}

		c1, err := net.ListenPacket(tt.net, tt.addr1)
		if err != nil {
			t.Fatalf("net.ListenPacket failed: %v", err)
		}
		c1.LocalAddr()
		c1.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c1.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c1.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
		defer c1.Close()

		c2, err := net.Dial(tt.net, c1.LocalAddr().String())
		if err != nil {
			t.Fatalf("net.Dial failed: %v", err)
		}
		c2.LocalAddr()
		c2.RemoteAddr()
		c2.SetDeadline(time.Now().Add(100 * time.Millisecond))
		c2.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
		c2.SetWriteDeadline(time.Now().Add(100 * time.Millisecond))
		defer c2.Close()

		if _, err := c2.Write(wb); err != nil {
			t.Fatalf("net.Conn.Write failed: %v", err)
		}
		rb1 := make([]byte, 128)
		if _, _, err := c1.ReadFrom(rb1); err != nil {
			t.Fatalf("net.PacetConn.ReadFrom failed: %v", err)
		}
		var dst net.Addr
		if netstr[0] == "ip" {
			dst = &net.IPAddr{IP: net.IPv4(127, 0, 0, 1)}
		} else {
			dst = c2.LocalAddr()
		}
		if _, err := c1.WriteTo(wb, dst); err != nil {
			t.Fatalf("net.PacketConn.WriteTo failed: %v", err)
		}
		rb2 := make([]byte, 128)
		if _, err := c2.Read(rb2); err != nil {
			t.Fatalf("net.Conn.Read failed: %v", err)
		}
	}
}
