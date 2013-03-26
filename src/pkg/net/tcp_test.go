// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"time"
)

func BenchmarkTCP4OneShot(b *testing.B) {
	benchmarkTCP(b, false, false, "127.0.0.1:0")
}

func BenchmarkTCP4OneShotTimeout(b *testing.B) {
	benchmarkTCP(b, false, true, "127.0.0.1:0")
}

func BenchmarkTCP4Persistent(b *testing.B) {
	benchmarkTCP(b, true, false, "127.0.0.1:0")
}

func BenchmarkTCP4PersistentTimeout(b *testing.B) {
	benchmarkTCP(b, true, true, "127.0.0.1:0")
}

func BenchmarkTCP6OneShot(b *testing.B) {
	if !supportsIPv6 {
		b.Skip("ipv6 is not supported")
	}
	benchmarkTCP(b, false, false, "[::1]:0")
}

func BenchmarkTCP6OneShotTimeout(b *testing.B) {
	if !supportsIPv6 {
		b.Skip("ipv6 is not supported")
	}
	benchmarkTCP(b, false, true, "[::1]:0")
}

func BenchmarkTCP6Persistent(b *testing.B) {
	if !supportsIPv6 {
		b.Skip("ipv6 is not supported")
	}
	benchmarkTCP(b, true, false, "[::1]:0")
}

func BenchmarkTCP6PersistentTimeout(b *testing.B) {
	if !supportsIPv6 {
		b.Skip("ipv6 is not supported")
	}
	benchmarkTCP(b, true, true, "[::1]:0")
}

func benchmarkTCP(b *testing.B, persistent, timeout bool, laddr string) {
	const msgLen = 512
	conns := b.N
	numConcurrent := runtime.GOMAXPROCS(-1) * 16
	msgs := 1
	if persistent {
		conns = numConcurrent
		msgs = b.N / conns
		if msgs == 0 {
			msgs = 1
		}
		if conns > b.N {
			conns = b.N
		}
	}
	sendMsg := func(c Conn, buf []byte) bool {
		n, err := c.Write(buf)
		if n != len(buf) || err != nil {
			b.Logf("Write failed: %v", err)
			return false
		}
		return true
	}
	recvMsg := func(c Conn, buf []byte) bool {
		for read := 0; read != len(buf); {
			n, err := c.Read(buf)
			read += n
			if err != nil {
				b.Logf("Read failed: %v", err)
				return false
			}
		}
		return true
	}
	ln, err := Listen("tcp", laddr)
	if err != nil {
		b.Fatalf("Listen failed: %v", err)
	}
	defer ln.Close()
	// Acceptor.
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				break
			}
			// Server connection.
			go func(c Conn) {
				defer c.Close()
				if timeout {
					c.SetDeadline(time.Now().Add(time.Hour)) // Not intended to fire.
				}
				var buf [msgLen]byte
				for m := 0; m < msgs; m++ {
					if !recvMsg(c, buf[:]) || !sendMsg(c, buf[:]) {
						break
					}
				}
			}(c)
		}
	}()
	sem := make(chan bool, numConcurrent)
	for i := 0; i < conns; i++ {
		sem <- true
		// Client connection.
		go func() {
			defer func() {
				<-sem
			}()
			c, err := Dial("tcp", ln.Addr().String())
			if err != nil {
				b.Logf("Dial failed: %v", err)
				return
			}
			defer c.Close()
			if timeout {
				c.SetDeadline(time.Now().Add(time.Hour)) // Not intended to fire.
			}
			var buf [msgLen]byte
			for m := 0; m < msgs; m++ {
				if !sendMsg(c, buf[:]) || !recvMsg(c, buf[:]) {
					break
				}
			}
		}()
	}
	for i := 0; i < cap(sem); i++ {
		sem <- true
	}
}

type resolveTCPAddrTest struct {
	net     string
	litAddr string
	addr    *TCPAddr
	err     error
}

var resolveTCPAddrTests = []resolveTCPAddrTest{
	{"tcp", "127.0.0.1:0", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil},
	{"tcp4", "127.0.0.1:65535", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 65535}, nil},

	{"tcp", "[::1]:1", &TCPAddr{IP: ParseIP("::1"), Port: 1}, nil},
	{"tcp6", "[::1]:65534", &TCPAddr{IP: ParseIP("::1"), Port: 65534}, nil},

	{"tcp", "[::1%en0]:1", &TCPAddr{IP: ParseIP("::1"), Port: 1, Zone: "en0"}, nil},
	{"tcp6", "[::1%911]:2", &TCPAddr{IP: ParseIP("::1"), Port: 2, Zone: "911"}, nil},

	{"", "127.0.0.1:0", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil}, // Go 1.0 behavior
	{"", "[::1]:0", &TCPAddr{IP: ParseIP("::1"), Port: 0}, nil},         // Go 1.0 behavior

	{"http", "127.0.0.1:0", nil, UnknownNetworkError("http")},
}

func init() {
	if ifi := loopbackInterface(); ifi != nil {
		index := fmt.Sprintf("%v", ifi.Index)
		resolveTCPAddrTests = append(resolveTCPAddrTests, []resolveTCPAddrTest{
			{"tcp6", "[fe80::1%" + ifi.Name + "]:3", &TCPAddr{IP: ParseIP("fe80::1"), Port: 3, Zone: zoneToString(ifi.Index)}, nil},
			{"tcp6", "[fe80::1%" + index + "]:4", &TCPAddr{IP: ParseIP("fe80::1"), Port: 4, Zone: index}, nil},
		}...)
	}
}

func TestResolveTCPAddr(t *testing.T) {
	for _, tt := range resolveTCPAddrTests {
		addr, err := ResolveTCPAddr(tt.net, tt.litAddr)
		if err != tt.err {
			t.Fatalf("ResolveTCPAddr(%v, %v) failed: %v", tt.net, tt.litAddr, err)
		}
		if !reflect.DeepEqual(addr, tt.addr) {
			t.Fatalf("got %#v; expected %#v", addr, tt.addr)
		}
	}
}

var tcpListenerNameTests = []struct {
	net   string
	laddr *TCPAddr
}{
	{"tcp4", &TCPAddr{IP: IPv4(127, 0, 0, 1)}},
	{"tcp4", &TCPAddr{}},
	{"tcp4", nil},
}

func TestTCPListenerName(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	for _, tt := range tcpListenerNameTests {
		ln, err := ListenTCP(tt.net, tt.laddr)
		if err != nil {
			t.Fatalf("ListenTCP failed: %v", err)
		}
		defer ln.Close()
		la := ln.Addr()
		if a, ok := la.(*TCPAddr); !ok || a.Port == 0 {
			t.Fatalf("got %v; expected a proper address with non-zero port number", la)
		}
	}
}

func TestIPv6LinkLocalUnicastTCP(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}
	ifi := loopbackInterface()
	if ifi == nil {
		t.Skip("loopback interface not found")
	}
	laddr := ipv6LinkLocalUnicastAddr(ifi)
	if laddr == "" {
		t.Skip("ipv6 unicast address on loopback not found")
	}

	type test struct {
		net, addr  string
		nameLookup bool
	}
	var tests = []test{
		{"tcp", "[" + laddr + "%" + ifi.Name + "]:0", false},
		{"tcp6", "[" + laddr + "%" + ifi.Name + "]:0", false},
	}
	switch runtime.GOOS {
	case "darwin", "freebsd", "opensbd", "netbsd":
		tests = append(tests, []test{
			{"tcp", "[localhost%" + ifi.Name + "]:0", true},
			{"tcp6", "[localhost%" + ifi.Name + "]:0", true},
		}...)
	case "linux":
		tests = append(tests, []test{
			{"tcp", "[ip6-localhost%" + ifi.Name + "]:0", true},
			{"tcp6", "[ip6-localhost%" + ifi.Name + "]:0", true},
		}...)
	}
	for _, tt := range tests {
		ln, err := Listen(tt.net, tt.addr)
		if err != nil {
			// It might return "LookupHost returned no
			// suitable address" error on some platforms.
			t.Logf("Listen failed: %v", err)
			continue
		}
		defer ln.Close()
		if la, ok := ln.Addr().(*TCPAddr); !ok || !tt.nameLookup && la.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", la)
		}

		done := make(chan int)
		go transponder(t, ln, done)

		c, err := Dial(tt.net, ln.Addr().String())
		if err != nil {
			t.Fatalf("Dial failed: %v", err)
		}
		defer c.Close()
		if la, ok := c.LocalAddr().(*TCPAddr); !ok || !tt.nameLookup && la.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", la)
		}
		if ra, ok := c.RemoteAddr().(*TCPAddr); !ok || !tt.nameLookup && ra.Zone == "" {
			t.Fatalf("got %v; expected a proper address with zone identifier", ra)
		}

		if _, err := c.Write([]byte("TCP OVER IPV6 LINKLOCAL TEST")); err != nil {
			t.Fatalf("Conn.Write failed: %v", err)
		}
		b := make([]byte, 32)
		if _, err := c.Read(b); err != nil {
			t.Fatalf("Conn.Read failed: %v", err)
		}

		<-done
	}
}
