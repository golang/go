// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"io"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Do not test empty datagrams by default.
// It causes unexplained timeouts on some systems,
// including Snow Leopard.  I think that the kernel
// doesn't quite expect them.
var testUDP = flag.Bool("udp", false, "whether to test UDP datagrams")

func runEcho(fd io.ReadWriter, done chan<- int) {
	var buf [1024]byte

	for {
		n, err := fd.Read(buf[0:])
		if err != nil || n == 0 || string(buf[:n]) == "END" {
			break
		}
		fd.Write(buf[0:n])
	}
	done <- 1
}

func runServe(t *testing.T, network, addr string, listening chan<- string, done chan<- int) {
	l, err := Listen(network, addr)
	if err != nil {
		t.Fatalf("net.Listen(%q, %q) = _, %v", network, addr, err)
	}
	listening <- l.Addr().String()

	for {
		fd, err := l.Accept()
		if err != nil {
			break
		}
		echodone := make(chan int)
		go runEcho(fd, echodone)
		<-echodone // make sure Echo stops
		l.Close()
	}
	done <- 1
}

func connect(t *testing.T, network, addr string, isEmpty bool) {
	var fd Conn
	var err error
	if network == "unixgram" {
		fd, err = DialUnix(network, &UnixAddr{addr + ".local", network}, &UnixAddr{addr, network})
	} else {
		fd, err = Dial(network, addr)
	}
	if err != nil {
		t.Fatalf("net.Dial(%q, %q) = _, %v", network, addr, err)
	}
	fd.SetReadDeadline(time.Now().Add(1 * time.Second))

	var b []byte
	if !isEmpty {
		b = []byte("hello, world\n")
	}
	var b1 [100]byte

	n, err1 := fd.Write(b)
	if n != len(b) {
		t.Fatalf("fd.Write(%q) = %d, %v", b, n, err1)
	}

	n, err1 = fd.Read(b1[0:])
	if n != len(b) || err1 != nil {
		t.Fatalf("fd.Read() = %d, %v (want %d, nil)", n, err1, len(b))
	}

	// Send explicit ending for unixpacket.
	// Older Linux kernels do not stop reads on close.
	if network == "unixpacket" {
		fd.Write([]byte("END"))
	}

	fd.Close()
}

func doTest(t *testing.T, network, listenaddr, dialaddr string) {
	t.Logf("Test %q %q %q", network, listenaddr, dialaddr)
	switch listenaddr {
	case "", "0.0.0.0", "[::]", "[::ffff:0.0.0.0]":
		if testing.Short() || avoidMacFirewall {
			t.Logf("skip wildcard listen during short test")
			return
		}
	}
	listening := make(chan string)
	done := make(chan int)
	if network == "tcp" || network == "tcp4" || network == "tcp6" {
		listenaddr += ":0" // any available port
	}
	go runServe(t, network, listenaddr, listening, done)
	addr := <-listening // wait for server to start
	if network == "tcp" || network == "tcp4" || network == "tcp6" {
		dialaddr += addr[strings.LastIndex(addr, ":"):]
	}
	connect(t, network, dialaddr, false)
	<-done // make sure server stopped
}

func TestTCPServer(t *testing.T) {
	doTest(t, "tcp", "", "127.0.0.1")
	doTest(t, "tcp", "0.0.0.0", "127.0.0.1")
	doTest(t, "tcp", "127.0.0.1", "127.0.0.1")
	doTest(t, "tcp4", "", "127.0.0.1")
	doTest(t, "tcp4", "0.0.0.0", "127.0.0.1")
	doTest(t, "tcp4", "127.0.0.1", "127.0.0.1")
	if supportsIPv6 {
		doTest(t, "tcp", "[::]", "[::1]")
		doTest(t, "tcp", "[::1]", "[::1]")
		doTest(t, "tcp6", "", "[::1]")
		doTest(t, "tcp6", "[::]", "[::1]")
		doTest(t, "tcp6", "[::1]", "[::1]")
	}
	if supportsIPv6 && supportsIPv4map {
		doTest(t, "tcp", "[::ffff:0.0.0.0]", "127.0.0.1")
		doTest(t, "tcp", "[::]", "127.0.0.1")
		doTest(t, "tcp4", "[::ffff:0.0.0.0]", "127.0.0.1")
		doTest(t, "tcp6", "", "127.0.0.1")
		doTest(t, "tcp6", "[::ffff:0.0.0.0]", "127.0.0.1")
		doTest(t, "tcp6", "[::]", "127.0.0.1")
		doTest(t, "tcp", "127.0.0.1", "[::ffff:127.0.0.1]")
		doTest(t, "tcp", "[::ffff:127.0.0.1]", "127.0.0.1")
		doTest(t, "tcp4", "127.0.0.1", "[::ffff:127.0.0.1]")
		doTest(t, "tcp4", "[::ffff:127.0.0.1]", "127.0.0.1")
		doTest(t, "tcp6", "127.0.0.1", "[::ffff:127.0.0.1]")
		doTest(t, "tcp6", "[::ffff:127.0.0.1]", "127.0.0.1")
	}
}

func TestUnixServer(t *testing.T) {
	// "unix" sockets are not supported on windows and Plan 9.
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		return
	}
	os.Remove("/tmp/gotest.net")
	doTest(t, "unix", "/tmp/gotest.net", "/tmp/gotest.net")
	os.Remove("/tmp/gotest.net")
	if runtime.GOOS == "linux" {
		doTest(t, "unixpacket", "/tmp/gotest.net", "/tmp/gotest.net")
		os.Remove("/tmp/gotest.net")
		// Test abstract unix domain socket, a Linux-ism
		doTest(t, "unix", "@gotest/net", "@gotest/net")
		doTest(t, "unixpacket", "@gotest/net", "@gotest/net")
	}
}

func runPacket(t *testing.T, network, addr string, listening chan<- string, done chan<- int) {
	c, err := ListenPacket(network, addr)
	if err != nil {
		t.Fatalf("net.ListenPacket(%q, %q) = _, %v", network, addr, err)
	}
	listening <- c.LocalAddr().String()
	var buf [1000]byte
Run:
	for {
		c.SetReadDeadline(time.Now().Add(10 * time.Millisecond))
		n, addr, err := c.ReadFrom(buf[0:])
		if e, ok := err.(Error); ok && e.Timeout() {
			select {
			case done <- 1:
				break Run
			default:
				continue Run
			}
		}
		if err != nil {
			break
		}
		if _, err = c.WriteTo(buf[0:n], addr); err != nil {
			t.Fatalf("WriteTo %v: %v", addr, err)
		}
	}
	c.Close()
	done <- 1
}

func doTestPacket(t *testing.T, network, listenaddr, dialaddr string, isEmpty bool) {
	t.Logf("TestPacket %q %q %q", network, listenaddr, dialaddr)
	listening := make(chan string)
	done := make(chan int)
	if network == "udp" {
		listenaddr += ":0" // any available port
	}
	go runPacket(t, network, listenaddr, listening, done)
	addr := <-listening // wait for server to start
	if network == "udp" {
		dialaddr += addr[strings.LastIndex(addr, ":"):]
	}
	connect(t, network, dialaddr, isEmpty)
	<-done // tell server to stop
	<-done // wait for stop
}

func TestUDPServer(t *testing.T) {
	if !*testUDP {
		return
	}
	for _, isEmpty := range []bool{false, true} {
		doTestPacket(t, "udp", "0.0.0.0", "127.0.0.1", isEmpty)
		doTestPacket(t, "udp", "", "127.0.0.1", isEmpty)
		if supportsIPv6 && supportsIPv4map {
			doTestPacket(t, "udp", "[::]", "[::ffff:127.0.0.1]", isEmpty)
			doTestPacket(t, "udp", "[::]", "127.0.0.1", isEmpty)
			doTestPacket(t, "udp", "0.0.0.0", "[::ffff:127.0.0.1]", isEmpty)
		}
	}
}

func TestUnixDatagramServer(t *testing.T) {
	// "unix" sockets are not supported on windows and Plan 9.
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		return
	}
	for _, isEmpty := range []bool{false} {
		os.Remove("/tmp/gotest1.net")
		os.Remove("/tmp/gotest1.net.local")
		doTestPacket(t, "unixgram", "/tmp/gotest1.net", "/tmp/gotest1.net", isEmpty)
		os.Remove("/tmp/gotest1.net")
		os.Remove("/tmp/gotest1.net.local")
		if runtime.GOOS == "linux" {
			// Test abstract unix domain socket, a Linux-ism
			doTestPacket(t, "unixgram", "@gotest1/net", "@gotest1/net", isEmpty)
		}
	}
}
