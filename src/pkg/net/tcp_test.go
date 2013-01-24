// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"runtime"
	"testing"
	"time"
)

func BenchmarkTCPOneShot(b *testing.B) {
	benchmarkTCP(b, false, false)
}

func BenchmarkTCPOneShotTimeout(b *testing.B) {
	benchmarkTCP(b, false, true)
}

func BenchmarkTCPPersistent(b *testing.B) {
	benchmarkTCP(b, true, false)
}

func BenchmarkTCPPersistentTimeout(b *testing.B) {
	benchmarkTCP(b, true, true)
}

func benchmarkTCP(b *testing.B, persistent, timeout bool) {
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
	ln, err := Listen("tcp", "127.0.0.1:0")
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

var resolveTCPAddrTests = []struct {
	net     string
	litAddr string
	addr    *TCPAddr
	err     error
}{
	{"tcp", "127.0.0.1:0", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil},
	{"tcp4", "127.0.0.1:65535", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 65535}, nil},

	{"tcp", "[::1]:1", &TCPAddr{IP: ParseIP("::1"), Port: 1}, nil},
	{"tcp6", "[::1]:65534", &TCPAddr{IP: ParseIP("::1"), Port: 65534}, nil},

	{"", "127.0.0.1:0", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 0}, nil}, // Go 1.0 behavior
	{"", "[::1]:0", &TCPAddr{IP: ParseIP("::1"), Port: 0}, nil},         // Go 1.0 behavior

	{"http", "127.0.0.1:0", nil, UnknownNetworkError("http")},
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
			t.Errorf("ListenTCP failed: %v", err)
			return
		}
		defer ln.Close()
		la := ln.Addr()
		if a, ok := la.(*TCPAddr); !ok || a.Port == 0 {
			t.Errorf("got %v; expected a proper address with non-zero port number", la)
			return
		}
	}
}
