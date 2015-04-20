// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"testing"
	"time"
)

var streamConnServerTests = []struct {
	snet  string // server side
	saddr string
	cnet  string // client side
	caddr string
	empty bool // test with empty data
}{
	{snet: "tcp", saddr: ":0", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "0.0.0.0:0", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]:0", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::]:0", cnet: "tcp", caddr: "::1"},

	{snet: "tcp", saddr: ":0", cnet: "tcp", caddr: "::1"},
	{snet: "tcp", saddr: "0.0.0.0:0", cnet: "tcp", caddr: "::1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]:0", cnet: "tcp", caddr: "::1"},
	{snet: "tcp", saddr: "[::]:0", cnet: "tcp", caddr: "127.0.0.1"},

	{snet: "tcp", saddr: ":0", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "0.0.0.0:0", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]:0", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::]:0", cnet: "tcp6", caddr: "::1"},

	{snet: "tcp", saddr: ":0", cnet: "tcp6", caddr: "::1"},
	{snet: "tcp", saddr: "0.0.0.0:0", cnet: "tcp6", caddr: "::1"},
	{snet: "tcp", saddr: "[::ffff:0.0.0.0]:0", cnet: "tcp6", caddr: "::1"},
	{snet: "tcp", saddr: "[::]:0", cnet: "tcp4", caddr: "127.0.0.1"},

	{snet: "tcp", saddr: "127.0.0.1:0", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::ffff:127.0.0.1]:0", cnet: "tcp", caddr: "127.0.0.1"},
	{snet: "tcp", saddr: "[::1]:0", cnet: "tcp", caddr: "::1"},

	{snet: "tcp4", saddr: ":0", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp4", saddr: "0.0.0.0:0", cnet: "tcp4", caddr: "127.0.0.1"},
	{snet: "tcp4", saddr: "[::ffff:0.0.0.0]:0", cnet: "tcp4", caddr: "127.0.0.1"},

	{snet: "tcp4", saddr: "127.0.0.1:0", cnet: "tcp4", caddr: "127.0.0.1"},

	{snet: "tcp6", saddr: ":0", cnet: "tcp6", caddr: "::1"},
	{snet: "tcp6", saddr: "[::]:0", cnet: "tcp6", caddr: "::1"},

	{snet: "tcp6", saddr: "[::1]:0", cnet: "tcp6", caddr: "::1"},

	{snet: "unix", saddr: testUnixAddr(), cnet: "unix", caddr: testUnixAddr()},
	{snet: "unix", saddr: "@gotest2/net", cnet: "unix", caddr: "@gotest2/net.local"},
}

func TestStreamConnServer(t *testing.T) {
	for _, tt := range streamConnServerTests {
		if !testableListenArgs(tt.snet, tt.saddr, tt.caddr) {
			t.Logf("skipping %s test", tt.snet+":"+tt.saddr+"->"+tt.caddr)
			continue
		}

		listening := make(chan string)
		done := make(chan int)
		switch tt.snet {
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
			taddr = JoinHostPort(tt.caddr, port)
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
	{net: "unixpacket", saddr: testUnixAddr(), caddr: testUnixAddr()},
	{net: "unixpacket", saddr: "@gotest4/net", caddr: "@gotest4/net.local"},
}

func TestSeqpacketConnServer(t *testing.T) {
	for _, tt := range seqpacketConnServerTests {
		if !testableListenArgs(tt.net, tt.saddr, tt.caddr) {
			t.Logf("skipping %s test", tt.net+":"+tt.saddr+"->"+tt.caddr)
			continue
		}
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

var datagramPacketConnServerTests = []struct {
	snet  string // server side
	saddr string
	cnet  string // client side
	caddr string
	dial  bool // test with Dial or DialUnix
	empty bool // test with empty data
}{
	{snet: "udp", saddr: ":0", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "0.0.0.0:0", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]:0", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::]:0", cnet: "udp", caddr: "::1"},

	{snet: "udp", saddr: ":0", cnet: "udp", caddr: "::1"},
	{snet: "udp", saddr: "0.0.0.0:0", cnet: "udp", caddr: "::1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]:0", cnet: "udp", caddr: "::1"},
	{snet: "udp", saddr: "[::]:0", cnet: "udp", caddr: "127.0.0.1"},

	{snet: "udp", saddr: ":0", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "0.0.0.0:0", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]:0", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::]:0", cnet: "udp6", caddr: "::1"},

	{snet: "udp", saddr: ":0", cnet: "udp6", caddr: "::1"},
	{snet: "udp", saddr: "0.0.0.0:0", cnet: "udp6", caddr: "::1"},
	{snet: "udp", saddr: "[::ffff:0.0.0.0]:0", cnet: "udp6", caddr: "::1"},
	{snet: "udp", saddr: "[::]:0", cnet: "udp4", caddr: "127.0.0.1"},

	{snet: "udp", saddr: "127.0.0.1:0", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::ffff:127.0.0.1]:0", cnet: "udp", caddr: "127.0.0.1"},
	{snet: "udp", saddr: "[::1]:0", cnet: "udp", caddr: "::1"},

	{snet: "udp4", saddr: ":0", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp4", saddr: "0.0.0.0:0", cnet: "udp4", caddr: "127.0.0.1"},
	{snet: "udp4", saddr: "[::ffff:0.0.0.0]:0", cnet: "udp4", caddr: "127.0.0.1"},

	{snet: "udp4", saddr: "127.0.0.1:0", cnet: "udp4", caddr: "127.0.0.1"},

	{snet: "udp6", saddr: ":0", cnet: "udp6", caddr: "::1"},
	{snet: "udp6", saddr: "[::]:0", cnet: "udp6", caddr: "::1"},

	{snet: "udp6", saddr: "[::1]:0", cnet: "udp6", caddr: "::1"},

	{snet: "udp", saddr: "127.0.0.1:0", cnet: "udp", caddr: "127.0.0.1", dial: true},
	{snet: "udp", saddr: "127.0.0.1:0", cnet: "udp", caddr: "127.0.0.1", empty: true},
	{snet: "udp", saddr: "127.0.0.1:0", cnet: "udp", caddr: "127.0.0.1", dial: true, empty: true},

	{snet: "udp", saddr: "[::1]:0", cnet: "udp", caddr: "::1", dial: true},
	{snet: "udp", saddr: "[::1]:0", cnet: "udp", caddr: "::1", empty: true},
	{snet: "udp", saddr: "[::1]:0", cnet: "udp", caddr: "::1", dial: true, empty: true},

	{snet: "unixgram", saddr: testUnixAddr(), cnet: "unixgram", caddr: testUnixAddr()},
	{snet: "unixgram", saddr: testUnixAddr(), cnet: "unixgram", caddr: testUnixAddr(), dial: true},
	{snet: "unixgram", saddr: testUnixAddr(), cnet: "unixgram", caddr: testUnixAddr(), empty: true},
	{snet: "unixgram", saddr: testUnixAddr(), cnet: "unixgram", caddr: testUnixAddr(), dial: true, empty: true},

	{snet: "unixgram", saddr: "@gotest6/net", cnet: "unixgram", caddr: "@gotest6/net.local"},
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
		t.Fatalf("ListenPacket(%q, %q) failed: %v", net, laddr, err)
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
