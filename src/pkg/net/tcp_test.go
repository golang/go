// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"io"
	"reflect"
	"runtime"
	"sync"
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
	numConcurrent := runtime.GOMAXPROCS(-1) * 2
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

func BenchmarkTCP4ConcurrentReadWrite(b *testing.B) {
	benchmarkTCPConcurrentReadWrite(b, "127.0.0.1:0")
}

func BenchmarkTCP6ConcurrentReadWrite(b *testing.B) {
	if !supportsIPv6 {
		b.Skip("ipv6 is not supported")
	}
	benchmarkTCPConcurrentReadWrite(b, "[::1]:0")
}

func benchmarkTCPConcurrentReadWrite(b *testing.B, laddr string) {
	// The benchmark creates GOMAXPROCS client/server pairs.
	// Each pair creates 4 goroutines: client reader/writer and server reader/writer.
	// The benchmark stresses concurrent reading and writing to the same connection.
	// Such pattern is used in net/http and net/rpc.

	b.StopTimer()

	P := runtime.GOMAXPROCS(0)
	N := b.N / P
	W := 1000

	// Setup P client/server connections.
	clients := make([]Conn, P)
	servers := make([]Conn, P)
	ln, err := Listen("tcp", laddr)
	if err != nil {
		b.Fatalf("Listen failed: %v", err)
	}
	defer ln.Close()
	done := make(chan bool)
	go func() {
		for p := 0; p < P; p++ {
			s, err := ln.Accept()
			if err != nil {
				b.Fatalf("Accept failed: %v", err)
			}
			servers[p] = s
		}
		done <- true
	}()
	for p := 0; p < P; p++ {
		c, err := Dial("tcp", ln.Addr().String())
		if err != nil {
			b.Fatalf("Dial failed: %v", err)
		}
		clients[p] = c
	}
	<-done

	b.StartTimer()

	var wg sync.WaitGroup
	wg.Add(4 * P)
	for p := 0; p < P; p++ {
		// Client writer.
		go func(c Conn) {
			defer wg.Done()
			var buf [1]byte
			for i := 0; i < N; i++ {
				v := byte(i)
				for w := 0; w < W; w++ {
					v *= v
				}
				buf[0] = v
				_, err := c.Write(buf[:])
				if err != nil {
					b.Fatalf("Write failed: %v", err)
				}
			}
		}(clients[p])

		// Pipe between server reader and server writer.
		pipe := make(chan byte, 128)

		// Server reader.
		go func(s Conn) {
			defer wg.Done()
			var buf [1]byte
			for i := 0; i < N; i++ {
				_, err := s.Read(buf[:])
				if err != nil {
					b.Fatalf("Read failed: %v", err)
				}
				pipe <- buf[0]
			}
		}(servers[p])

		// Server writer.
		go func(s Conn) {
			defer wg.Done()
			var buf [1]byte
			for i := 0; i < N; i++ {
				v := <-pipe
				for w := 0; w < W; w++ {
					v *= v
				}
				buf[0] = v
				_, err := s.Write(buf[:])
				if err != nil {
					b.Fatalf("Write failed: %v", err)
				}
			}
			s.Close()
		}(servers[p])

		// Client reader.
		go func(c Conn) {
			defer wg.Done()
			var buf [1]byte
			for i := 0; i < N; i++ {
				_, err := c.Read(buf[:])
				if err != nil {
					b.Fatalf("Read failed: %v", err)
				}
			}
			c.Close()
		}(clients[p])
	}
	wg.Wait()
}

type resolveTCPAddrTest struct {
	net           string
	litAddrOrName string
	addr          *TCPAddr
	err           error
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

	{"tcp", ":12345", &TCPAddr{Port: 12345}, nil},

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
	if ips, err := LookupIP("localhost"); err == nil && len(ips) > 1 && supportsIPv4 && supportsIPv6 {
		resolveTCPAddrTests = append(resolveTCPAddrTests, []resolveTCPAddrTest{
			{"tcp", "localhost:5", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 5}, nil},
			{"tcp4", "localhost:6", &TCPAddr{IP: IPv4(127, 0, 0, 1), Port: 6}, nil},
			{"tcp6", "localhost:7", &TCPAddr{IP: IPv6loopback, Port: 7}, nil},
		}...)
	}
}

func TestResolveTCPAddr(t *testing.T) {
	for _, tt := range resolveTCPAddrTests {
		addr, err := ResolveTCPAddr(tt.net, tt.litAddrOrName)
		if err != tt.err {
			t.Fatalf("ResolveTCPAddr(%q, %q) failed: %v", tt.net, tt.litAddrOrName, err)
		}
		if !reflect.DeepEqual(addr, tt.addr) {
			t.Fatalf("ResolveTCPAddr(%q, %q) = %#v, want %#v", tt.net, tt.litAddrOrName, addr, tt.addr)
		}
		if err == nil {
			str := addr.String()
			addr1, err := ResolveTCPAddr(tt.net, str)
			if err != nil {
				t.Fatalf("ResolveTCPAddr(%q, %q) [from %q]: %v", tt.net, str, tt.litAddrOrName, err)
			}
			if !reflect.DeepEqual(addr1, addr) {
				t.Fatalf("ResolveTCPAddr(%q, %q) [from %q] = %#v, want %#v", tt.net, str, tt.litAddrOrName, addr1, addr)
			}
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

func TestTCPConcurrentAccept(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	const N = 10
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			for {
				c, err := ln.Accept()
				if err != nil {
					break
				}
				c.Close()
			}
			wg.Done()
		}()
	}
	for i := 0; i < 10*N; i++ {
		c, err := Dial("tcp", ln.Addr().String())
		if err != nil {
			t.Fatalf("Dial failed: %v", err)
		}
		c.Close()
	}
	ln.Close()
	wg.Wait()
}

func TestTCPReadWriteMallocs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer ln.Close()
	var server Conn
	errc := make(chan error)
	go func() {
		var err error
		server, err = ln.Accept()
		errc <- err
	}()
	client, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("Dial failed: %v", err)
	}
	if err := <-errc; err != nil {
		t.Fatalf("Accept failed: %v", err)
	}
	defer server.Close()
	var buf [128]byte
	mallocs := testing.AllocsPerRun(1000, func() {
		_, err := server.Write(buf[:])
		if err != nil {
			t.Fatalf("Write failed: %v", err)
		}
		_, err = io.ReadFull(client, buf[:])
		if err != nil {
			t.Fatalf("Read failed: %v", err)
		}
	})
	if mallocs > 0 {
		t.Fatalf("Got %v allocs, want 0", mallocs)
	}
}

func TestTCPStress(t *testing.T) {
	const conns = 2
	const msgLen = 512
	msgs := int(1e4)
	if testing.Short() {
		msgs = 1e2
	}

	sendMsg := func(c Conn, buf []byte) bool {
		n, err := c.Write(buf)
		if n != len(buf) || err != nil {
			t.Logf("Write failed: %v", err)
			return false
		}
		return true
	}
	recvMsg := func(c Conn, buf []byte) bool {
		for read := 0; read != len(buf); {
			n, err := c.Read(buf)
			read += n
			if err != nil {
				t.Logf("Read failed: %v", err)
				return false
			}
		}
		return true
	}

	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
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
				var buf [msgLen]byte
				for m := 0; m < msgs; m++ {
					if !recvMsg(c, buf[:]) || !sendMsg(c, buf[:]) {
						break
					}
				}
			}(c)
		}
	}()
	done := make(chan bool)
	for i := 0; i < conns; i++ {
		// Client connection.
		go func() {
			defer func() {
				done <- true
			}()
			c, err := Dial("tcp", ln.Addr().String())
			if err != nil {
				t.Logf("Dial failed: %v", err)
				return
			}
			defer c.Close()
			var buf [msgLen]byte
			for m := 0; m < msgs; m++ {
				if !sendMsg(c, buf[:]) || !recvMsg(c, buf[:]) {
					break
				}
			}
		}()
	}
	for i := 0; i < conns; i++ {
		<-done
	}
}
