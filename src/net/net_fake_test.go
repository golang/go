// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js || wasip1

package net

// GOOS=js and GOOS=wasip1 do not have typical socket networking capabilities
// found on other platforms. To help run test suites of the stdlib packages,
// an in-memory "fake network" facility is implemented.
//
// The tests in this files are intended to validate the behavior of the fake
// network stack on these platforms.

import "testing"

func TestFakeConn(t *testing.T) {
	tests := []struct {
		name   string
		listen func() (Listener, error)
		dial   func(Addr) (Conn, error)
		addr   func(*testing.T, Addr)
	}{
		{
			name: "Listener:tcp",
			listen: func() (Listener, error) {
				return Listen("tcp", ":0")
			},
			dial: func(addr Addr) (Conn, error) {
				return Dial(addr.Network(), addr.String())
			},
			addr: testFakeTCPAddr,
		},

		{
			name: "ListenTCP:tcp",
			listen: func() (Listener, error) {
				// Creating a listening TCP connection with a nil address must
				// select an IP address on localhost with a random port.
				// This test verifies that the fake network facility does that.
				return ListenTCP("tcp", nil)
			},
			dial: func(addr Addr) (Conn, error) {
				// Connecting a listening TCP connection will select a local
				// address on the local network and connects to the destination
				// address.
				return DialTCP("tcp", nil, addr.(*TCPAddr))
			},
			addr: testFakeTCPAddr,
		},

		{
			name: "ListenUnix:unix",
			listen: func() (Listener, error) {
				return ListenUnix("unix", &UnixAddr{Name: "test"})
			},
			dial: func(addr Addr) (Conn, error) {
				return DialUnix("unix", nil, addr.(*UnixAddr))
			},
			addr: testFakeUnixAddr("unix", "test"),
		},

		{
			name: "ListenUnix:unixpacket",
			listen: func() (Listener, error) {
				return ListenUnix("unixpacket", &UnixAddr{Name: "test"})
			},
			dial: func(addr Addr) (Conn, error) {
				return DialUnix("unixpacket", nil, addr.(*UnixAddr))
			},
			addr: testFakeUnixAddr("unixpacket", "test"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			l, err := test.listen()
			if err != nil {
				t.Fatal(err)
			}
			defer l.Close()
			test.addr(t, l.Addr())

			c, err := test.dial(l.Addr())
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			test.addr(t, c.LocalAddr())
			test.addr(t, c.RemoteAddr())
		})
	}
}

func TestFakePacketConn(t *testing.T) {
	tests := []struct {
		name   string
		listen func() (PacketConn, error)
		dial   func(Addr) (Conn, error)
		addr   func(*testing.T, Addr)
	}{
		{
			name: "ListenPacket:udp",
			listen: func() (PacketConn, error) {
				return ListenPacket("udp", ":0")
			},
			dial: func(addr Addr) (Conn, error) {
				return Dial(addr.Network(), addr.String())
			},
			addr: testFakeUDPAddr,
		},

		{
			name: "ListenUDP:udp",
			listen: func() (PacketConn, error) {
				// Creating a listening UDP connection with a nil address must
				// select an IP address on localhost with a random port.
				// This test verifies that the fake network facility does that.
				return ListenUDP("udp", nil)
			},
			dial: func(addr Addr) (Conn, error) {
				// Connecting a listening UDP connection will select a local
				// address on the local network and connects to the destination
				// address.
				return DialUDP("udp", nil, addr.(*UDPAddr))
			},
			addr: testFakeUDPAddr,
		},

		{
			name: "ListenUnixgram:unixgram",
			listen: func() (PacketConn, error) {
				return ListenUnixgram("unixgram", &UnixAddr{Name: "test"})
			},
			dial: func(addr Addr) (Conn, error) {
				return DialUnix("unixgram", nil, addr.(*UnixAddr))
			},
			addr: testFakeUnixAddr("unixgram", "test"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			l, err := test.listen()
			if err != nil {
				t.Fatal(err)
			}
			defer l.Close()
			test.addr(t, l.LocalAddr())

			c, err := test.dial(l.LocalAddr())
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			test.addr(t, c.LocalAddr())
			test.addr(t, c.RemoteAddr())
		})
	}
}

func testFakeTCPAddr(t *testing.T, addr Addr) {
	t.Helper()
	if a, ok := addr.(*TCPAddr); !ok {
		t.Errorf("Addr is not *TCPAddr: %T", addr)
	} else {
		testFakeNetAddr(t, a.IP, a.Port)
	}
}

func testFakeUDPAddr(t *testing.T, addr Addr) {
	t.Helper()
	if a, ok := addr.(*UDPAddr); !ok {
		t.Errorf("Addr is not *UDPAddr: %T", addr)
	} else {
		testFakeNetAddr(t, a.IP, a.Port)
	}
}

func testFakeNetAddr(t *testing.T, ip IP, port int) {
	t.Helper()
	if port == 0 {
		t.Error("network address is missing port")
	} else if len(ip) == 0 {
		t.Error("network address is missing IP")
	} else if !ip.Equal(IPv4(127, 0, 0, 1)) {
		t.Errorf("network address has wrong IP: %s", ip)
	}
}

func testFakeUnixAddr(net, name string) func(*testing.T, Addr) {
	return func(t *testing.T, addr Addr) {
		t.Helper()
		if a, ok := addr.(*UnixAddr); !ok {
			t.Errorf("Addr is not *UnixAddr: %T", addr)
		} else if a.Net != net {
			t.Errorf("unix address has wrong net: want=%q got=%q", net, a.Net)
		} else if a.Name != name {
			t.Errorf("unix address has wrong name: want=%q got=%q", name, a.Name)
		}
	}
}
