// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"io"
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"
)

func skipServerTest(net, unixsotype, addr string, ipv6, ipv4map, linuxonly bool) bool {
	switch runtime.GOOS {
	case "linux":
	case "plan9", "windows":
		// "unix" sockets are not supported on Windows and Plan 9.
		if net == unixsotype {
			return true
		}
	default:
		if net == unixsotype && linuxonly {
			return true
		}
	}
	switch addr {
	case "", "0.0.0.0", "[::ffff:0.0.0.0]", "[::]":
		if testing.Short() || !*testExternal {
			return true
		}
	}
	if ipv6 && !supportsIPv6 {
		return true
	}
	if ipv4map && !supportsIPv4map {
		return true
	}
	return false
}

func tempfile(filename string) string {
	// use /tmp in case it is prohibited to create
	// UNIX sockets in TMPDIR
	return "/tmp/" + filename + "." + strconv.Itoa(os.Getpid())
}

var streamConnServerTests = []struct {
	snet    string // server side
	saddr   string
	cnet    string // client side
	caddr   string
	ipv6    bool // test with underlying AF_INET6 socket
	ipv4map bool // test with IPv6 IPv4-mapping functionality
	empty   bool // test with empty data
	linux   bool // test with abstract unix domain socket, a Linux-ism
}{
	{snet: "tcp", saddr: "", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "0.0.0.0", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::]", cnet: "tcp", caddr: "[::1]", ipv6: true},

	{snet: "tcp", saddr: "", cnet: "tcp", caddr: "[::1]", ipv4map: true},
	{snet: "tcp", saddr: "0.0.0.0", cnet: "tcp", caddr: "[::1]", ipv4map: true},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]", cnet: "tcp", caddr: "[::1]", ipv4map: true},
	{snet: "tcp", saddr: "[::]", cnet: "tcp", caddr: "127.0.0.1", ipv4map: true},

	{snet: "tcp", saddr: "", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "0.0.0.0", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::]", cnet: "tcp6", caddr: "[::1]", ipv6: true},

	{snet: "tcp", saddr: "", cnet: "tcp6", caddr: "[::1]", ipv4map: true},
	{snet: "tcp", saddr: "0.0.0.0", cnet: "tcp6", caddr: "[::1]", ipv4map: true},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]", cnet: "tcp6", caddr: "[::1]", ipv4map: true},
	{snet: "tcp", saddr: "[::]", cnet: "tcp4", caddr: "127.0.0.1", ipv4map: true},

	{snet: "tcp", saddr: "127.0.0.1", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:127.0.0.1]", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::1]", cnet: "tcp", caddr: "[::1]", ipv6: true},

	{snet: "tcp4", saddr: "", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp4", saddr: "0.0.0.0", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp4", saddr: "[::ffff:0.0.0.0]", cnet: "tcp4", caddr: "127.0.0.1"},

	{snet: "tcp4", saddr: "127.0.0.1", cnet: "tcp4", caddr: "127.0.0.1"},

	{snet: "tcp6", saddr: "", cnet: "tcp6", caddr: "[::1]", ipv6: true},
	{snet: "tcp6", saddr: "[::]", cnet: "tcp6", caddr: "[::1]", ipv6: true},

	{snet: "tcp6", saddr: "[::1]", cnet: "tcp6", caddr: "[::1]", ipv6: true},

	{snet: "unix", saddr: tempfile("gotest1.net"), cnet: "unix", caddr: tempfile("gotest1.net.local")},
	{snet: "unix", saddr: "@gotest2/net", cnet: "unix", caddr: "@gotest2/net.local", linux: true},
}

func TestStreamConnServer(t *testing.T) {
	for _, tt := range streamConnServerTests {
		if skipServerTest(tt.snet, "unix", tt.saddr, tt.ipv6, tt.ipv4map, tt.linux) {
			continue
		}

		listening := make(chan string)
		done := make(chan int)
		switch tt.snet {
		case "tcp", "tcp4", "tcp6":
			tt.saddr += ":0"
		case "unix":
			os.Remove(tt.saddr)
			os.Remove(tt.caddr)
		}

		go runStreamConnServer(t, tt.snet, tt.saddr, listening, done)
		taddr := <-listening // wait for server to start

		switch tt.cnet {
		case "tcp", "tcp4", "tcp6":
			_, port, err := SplitHostPort(taddr)
			if err != nil {
				t.Fatalf("SplitHostPort(%q) failed: %v", taddr, err)
			}
			taddr = tt.caddr + ":" + port
		}

		runStreamConnClient(t, tt.cnet, taddr, tt.empty)
		<-done // make sure server stopped

		switch tt.snet {
		case "unix":
			os.Remove(tt.saddr)
			os.Remove(tt.caddr)
		}
	}
}

var seqpacketConnServerTests = []struct {
	net   string
	saddr string // server address
	caddr string // client address
	empty bool   // test with empty data
}{
	{net: "unixpacket", saddr: tempfile("/gotest3.net"), caddr: tempfile("gotest3.net.local")},
	{net: "unixpacket", saddr: "@gotest4/net", caddr: "@gotest4/net.local"},
}

func TestSeqpacketConnServer(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	for _, tt := range seqpacketConnServerTests {
		listening := make(chan string)
		done := make(chan int)
		switch tt.net {
		case "unixpacket":
			os.Remove(tt.saddr)
			os.Remove(tt.caddr)
		}

		go runStreamConnServer(t, tt.net, tt.saddr, listening, done)
		taddr := <-listening // wait for server to start

		runStreamConnClient(t, tt.net, taddr, tt.empty)
		<-done // make sure server stopped

		switch tt.net {
		case "unixpacket":
			os.Remove(tt.saddr)
			os.Remove(tt.caddr)
		}
	}
}

func runStreamConnServer(t *testing.T, net, laddr string, listening chan<- string, done chan<- int) {
	defer close(done)
	l, err := Listen(net, laddr)
	if err != nil {
		t.Errorf("Listen(%q, %q) failed: %v", net, laddr, err)
		listening <- "<nil>"
		return
	}
	defer l.Close()
	listening <- l.Addr().String()

	echo := func(rw io.ReadWriter, done chan<- int) {
		buf := make([]byte, 1024)
		for {
			n, err := rw.Read(buf[0:])
			if err != nil || n == 0 || string(buf[:n]) == "END" {
				break
			}
			rw.Write(buf[0:n])
		}
		close(done)
	}

run:
	for {
		c, err := l.Accept()
		if err != nil {
			t.Logf("Accept failed: %v", err)
			continue run
		}
		echodone := make(chan int)
		go echo(c, echodone)
		<-echodone // make sure echo stopped
		c.Close()
		break run
	}
}

func runStreamConnClient(t *testing.T, net, taddr string, isEmpty bool) {
	c, err := Dial(net, taddr)
	if err != nil {
		t.Fatalf("Dial(%q, %q) failed: %v", net, taddr, err)
	}
	defer c.Close()
	c.SetReadDeadline(time.Now().Add(1 * time.Second))

	var wb []byte
	if !isEmpty {
		wb = []byte("StreamConnClient by Dial\n")
	}
	if n, err := c.Write(wb); err != nil || n != len(wb) {
		t.Fatalf("Write failed: %v, %v; want %v, <nil>", n, err, len(wb))
	}

	rb := make([]byte, 1024)
	if n, err := c.Read(rb[0:]); err != nil || n != len(wb) {
		t.Fatalf("Read failed: %v, %v; want %v, <nil>", n, err, len(wb))
	}

	// Send explicit ending for unixpacket.
	// Older Linux kernels do not stop reads on close.
	switch net {
	case "unixpacket":
		c.Write([]byte("END"))
	}
}

// Do not test empty datagrams by default.
// It causes unexplained timeouts on some systems,
// including Snow Leopard.  I think that the kernel
// doesn't quite expect them.
var testDatagram = flag.Bool("datagram", false, "whether to test udp and unixgram")

var datagramPacketConnServerTests = []struct {
	snet    string // server side
	saddr   string
	cnet    string // client side
	caddr   string
	ipv6    bool // test with underlying AF_INET6 socket
	ipv4map bool // test with IPv6 IPv4-mapping functionality
	dial    bool // test with Dial or DialUnix
	empty   bool // test with empty data
	linux   bool // test with abstract unix domain socket, a Linux-ism
}{
	{snet: "udp", saddr: "", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "0.0.0.0", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::]", cnet: "udp", caddr: "[::1]", ipv6: true},

	{snet: "udp", saddr: "", cnet: "udp", caddr: "[::1]", ipv4map: true},
	{snet: "udp", saddr: "0.0.0.0", cnet: "udp", caddr: "[::1]", ipv4map: true},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]", cnet: "udp", caddr: "[::1]", ipv4map: true},
	{snet: "udp", saddr: "[::]", cnet: "udp", caddr: "127.0.0.1", ipv4map: true},

	{snet: "udp", saddr: "", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "0.0.0.0", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::]", cnet: "udp6", caddr: "[::1]", ipv6: true},

	{snet: "udp", saddr: "", cnet: "udp6", caddr: "[::1]", ipv4map: true},
	{snet: "udp", saddr: "0.0.0.0", cnet: "udp6", caddr: "[::1]", ipv4map: true},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]", cnet: "udp6", caddr: "[::1]", ipv4map: true},
	{snet: "udp", saddr: "[::]", cnet: "udp4", caddr: "127.0.0.1", ipv4map: true},

	{snet: "udp", saddr: "127.0.0.1", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:127.0.0.1]", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::1]", cnet: "udp", caddr: "[::1]", ipv6: true},

	{snet: "udp4", saddr: "", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp4", saddr: "0.0.0.0", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp4", saddr: "[::ffff:0.0.0.0]", cnet: "udp4", caddr: "127.0.0.1"},

	{snet: "udp4", saddr: "127.0.0.1", cnet: "udp4", caddr: "127.0.0.1"},

	{snet: "udp6", saddr: "", cnet: "udp6", caddr: "[::1]", ipv6: true},
	{snet: "udp6", saddr: "[::]", cnet: "udp6", caddr: "[::1]", ipv6: true},

	{snet: "udp6", saddr: "[::1]", cnet: "udp6", caddr: "[::1]", ipv6: true},

	{snet: "udp", saddr: "127.0.0.1", cnet: "udp", caddr: "127.0.0.1", dial: true},
	{snet: "udp", saddr: "127.0.0.1", cnet: "udp", caddr: "127.0.0.1", empty: true},
	{snet: "udp", saddr: "127.0.0.1", cnet: "udp", caddr: "127.0.0.1", dial: true, empty: true},

	{snet: "udp", saddr: "[::1]", cnet: "udp", caddr: "[::1]", ipv6: true, dial: true},
	{snet: "udp", saddr: "[::1]", cnet: "udp", caddr: "[::1]", ipv6: true, empty: true},
	{snet: "udp", saddr: "[::1]", cnet: "udp", caddr: "[::1]", ipv6: true, dial: true, empty: true},

	{snet: "unixgram", saddr: tempfile("gotest5.net"), cnet: "unixgram", caddr: tempfile("gotest5.net.local")},
	{snet: "unixgram", saddr: tempfile("gotest5.net"), cnet: "unixgram", caddr: tempfile("gotest5.net.local"), dial: true},
	{snet: "unixgram", saddr: tempfile("gotest5.net"), cnet: "unixgram", caddr: tempfile("gotest5.net.local"), empty: true},
	{snet: "unixgram", saddr: tempfile("gotest5.net"), cnet: "unixgram", caddr: tempfile("gotest5.net.local"), dial: true, empty: true},

	{snet: "unixgram", saddr: "@gotest6/net", cnet: "unixgram", caddr: "@gotest6/net.local", linux: true},
}

func TestDatagramPacketConnServer(t *testing.T) {
	if !*testDatagram {
		return
	}

	for _, tt := range datagramPacketConnServerTests {
		if skipServerTest(tt.snet, "unixgram", tt.saddr, tt.ipv6, tt.ipv4map, tt.linux) {
			continue
		}

		listening := make(chan string)
		done := make(chan int)
		switch tt.snet {
		case "udp", "udp4", "udp6":
			tt.saddr += ":0"
		case "unixgram":
			os.Remove(tt.saddr)
			os.Remove(tt.caddr)
		}

		go runDatagramPacketConnServer(t, tt.snet, tt.saddr, listening, done)
		taddr := <-listening // wait for server to start

		switch tt.cnet {
		case "udp", "udp4", "udp6":
			_, port, err := SplitHostPort(taddr)
			if err != nil {
				t.Fatalf("SplitHostPort(%q) failed: %v", taddr, err)
			}
			taddr = tt.caddr + ":" + port
			tt.caddr += ":0"
		}
		if tt.dial {
			runDatagramConnClient(t, tt.cnet, tt.caddr, taddr, tt.empty)
		} else {
			runDatagramPacketConnClient(t, tt.cnet, tt.caddr, taddr, tt.empty)
		}
		<-done // tell server to stop
		<-done // make sure server stopped

		switch tt.snet {
		case "unixgram":
			os.Remove(tt.saddr)
			os.Remove(tt.caddr)
		}
	}
}

func runDatagramPacketConnServer(t *testing.T, net, laddr string, listening chan<- string, done chan<- int) {
	c, err := ListenPacket(net, laddr)
	if err != nil {
		t.Errorf("ListenPacket(%q, %q) failed: %v", net, laddr, err)
		listening <- "<nil>"
		done <- 1
		return
	}
	defer c.Close()
	listening <- c.LocalAddr().String()

	buf := make([]byte, 1024)
run:
	for {
		c.SetReadDeadline(time.Now().Add(10 * time.Millisecond))
		n, ra, err := c.ReadFrom(buf[0:])
		if nerr, ok := err.(Error); ok && nerr.Timeout() {
			select {
			case done <- 1:
				break run
			default:
				continue run
			}
		}
		if err != nil {
			break run
		}
		if _, err = c.WriteTo(buf[0:n], ra); err != nil {
			t.Errorf("WriteTo(%v) failed: %v", ra, err)
			break run
		}
	}
	done <- 1
}

func runDatagramConnClient(t *testing.T, net, laddr, taddr string, isEmpty bool) {
	var c Conn
	var err error
	switch net {
	case "udp", "udp4", "udp6":
		c, err = Dial(net, taddr)
		if err != nil {
			t.Fatalf("Dial(%q, %q) failed: %v", net, taddr, err)
		}
	case "unixgram":
		c, err = DialUnix(net, &UnixAddr{Name: laddr, Net: net}, &UnixAddr{Name: taddr, Net: net})
		if err != nil {
			t.Fatalf("DialUnix(%q, {%q, %q}) failed: %v", net, laddr, taddr, err)
		}
	}
	defer c.Close()
	c.SetReadDeadline(time.Now().Add(1 * time.Second))

	var wb []byte
	if !isEmpty {
		wb = []byte("DatagramConnClient by Dial\n")
	}
	if n, err := c.Write(wb[0:]); err != nil || n != len(wb) {
		t.Fatalf("Write failed: %v, %v; want %v, <nil>", n, err, len(wb))
	}

	rb := make([]byte, 1024)
	if n, err := c.Read(rb[0:]); err != nil || n != len(wb) {
		t.Fatalf("Read failed: %v, %v; want %v, <nil>", n, err, len(wb))
	}
}

func runDatagramPacketConnClient(t *testing.T, net, laddr, taddr string, isEmpty bool) {
	var ra Addr
	var err error
	switch net {
	case "udp", "udp4", "udp6":
		ra, err = ResolveUDPAddr(net, taddr)
		if err != nil {
			t.Fatalf("ResolveUDPAddr(%q, %q) failed: %v", net, taddr, err)
		}
	case "unixgram":
		ra, err = ResolveUnixAddr(net, taddr)
		if err != nil {
			t.Fatalf("ResolveUxixAddr(%q, %q) failed: %v", net, taddr, err)
		}
	}
	c, err := ListenPacket(net, laddr)
	if err != nil {
		t.Fatalf("ListenPacket(%q, %q) faild: %v", net, laddr, err)
	}
	defer c.Close()
	c.SetReadDeadline(time.Now().Add(1 * time.Second))

	var wb []byte
	if !isEmpty {
		wb = []byte("DatagramPacketConnClient by ListenPacket\n")
	}
	if n, err := c.WriteTo(wb[0:], ra); err != nil || n != len(wb) {
		t.Fatalf("WriteTo(%v) failed: %v, %v; want %v, <nil>", ra, n, err, len(wb))
	}

	rb := make([]byte, 1024)
	if n, _, err := c.ReadFrom(rb[0:]); err != nil || n != len(wb) {
		t.Fatalf("ReadFrom failed: %v, %v; want %v, <nil>", n, err, len(wb))
	}
}
