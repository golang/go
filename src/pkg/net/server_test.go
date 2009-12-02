// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io";
	"os";
	"strings";
	"syscall";
	"testing";
)

func runEcho(fd io.ReadWriter, done chan<- int) {
	var buf [1024]byte;

	for {
		n, err := fd.Read(&buf);
		if err != nil || n == 0 {
			break
		}
		fd.Write(buf[0:n]);
	}
	done <- 1;
}

func runServe(t *testing.T, network, addr string, listening chan<- string, done chan<- int) {
	l, err := Listen(network, addr);
	if err != nil {
		t.Fatalf("net.Listen(%q, %q) = _, %v", network, addr, err)
	}
	listening <- l.Addr().String();

	for {
		fd, err := l.Accept();
		if err != nil {
			break
		}
		echodone := make(chan int);
		go runEcho(fd, echodone);
		<-echodone;	// make sure Echo stops
		l.Close();
	}
	done <- 1;
}

func connect(t *testing.T, network, addr string, isEmpty bool) {
	var laddr string;
	if network == "unixgram" {
		laddr = addr + ".local"
	}
	fd, err := Dial(network, laddr, addr);
	if err != nil {
		t.Fatalf("net.Dial(%q, %q, %q) = _, %v", network, laddr, addr, err)
	}
	fd.SetReadTimeout(10e6);	// 10ms

	var b []byte;
	if !isEmpty {
		b = strings.Bytes("hello, world\n")
	}
	var b1 [100]byte;

	n, err := fd.Write(b);
	if n != len(b) {
		t.Fatalf("fd.Write(%q) = %d, %v", b, n, err)
	}

	n, err = fd.Read(&b1);
	if n != len(b) || err != nil {
		t.Fatalf("fd.Read() = %d, %v (want %d, nil)", n, err, len(b))
	}
	fd.Close();
}

func doTest(t *testing.T, network, listenaddr, dialaddr string) {
	t.Logf("Test %s %s %s\n", network, listenaddr, dialaddr);
	listening := make(chan string);
	done := make(chan int);
	if network == "tcp" {
		listenaddr += ":0"	// any available port
	}
	go runServe(t, network, listenaddr, listening, done);
	addr := <-listening;	// wait for server to start
	if network == "tcp" {
		dialaddr += addr[strings.LastIndex(addr, ":"):]
	}
	connect(t, network, dialaddr, false);
	<-done;	// make sure server stopped
}

func TestTCPServer(t *testing.T) {
	doTest(t, "tcp", "0.0.0.0", "127.0.0.1");
	doTest(t, "tcp", "", "127.0.0.1");
	if kernelSupportsIPv6() {
		doTest(t, "tcp", "[::]", "[::ffff:127.0.0.1]");
		doTest(t, "tcp", "[::]", "127.0.0.1");
		doTest(t, "tcp", "0.0.0.0", "[::ffff:127.0.0.1]");
	}
}

func TestUnixServer(t *testing.T) {
	os.Remove("/tmp/gotest.net");
	doTest(t, "unix", "/tmp/gotest.net", "/tmp/gotest.net");
	os.Remove("/tmp/gotest.net");
	if syscall.OS == "linux" {
		// Test abstract unix domain socket, a Linux-ism
		doTest(t, "unix", "@gotest/net", "@gotest/net")
	}
}

func runPacket(t *testing.T, network, addr string, listening chan<- string, done chan<- int) {
	c, err := ListenPacket(network, addr);
	if err != nil {
		t.Fatalf("net.ListenPacket(%q, %q) = _, %v", network, addr, err)
	}
	listening <- c.LocalAddr().String();
	c.SetReadTimeout(10e6);	// 10ms
	var buf [1000]byte;
	for {
		n, addr, err := c.ReadFrom(&buf);
		if isEAGAIN(err) {
			if done <- 1 {
				break
			}
			continue;
		}
		if err != nil {
			break
		}
		if _, err = c.WriteTo(buf[0:n], addr); err != nil {
			t.Fatalf("WriteTo %v: %v", addr, err)
		}
	}
	c.Close();
	done <- 1;
}

func doTestPacket(t *testing.T, network, listenaddr, dialaddr string, isEmpty bool) {
	t.Logf("TestPacket %s %s %s\n", network, listenaddr, dialaddr);
	listening := make(chan string);
	done := make(chan int);
	if network == "udp" {
		listenaddr += ":0"	// any available port
	}
	go runPacket(t, network, listenaddr, listening, done);
	addr := <-listening;	// wait for server to start
	if network == "udp" {
		dialaddr += addr[strings.LastIndex(addr, ":"):]
	}
	connect(t, network, dialaddr, isEmpty);
	<-done;	// tell server to stop
	<-done;	// wait for stop
}

func TestUDPServer(t *testing.T) {
	for _, isEmpty := range []bool{false, true} {
		doTestPacket(t, "udp", "0.0.0.0", "127.0.0.1", isEmpty);
		doTestPacket(t, "udp", "", "127.0.0.1", isEmpty);
		if kernelSupportsIPv6() {
			doTestPacket(t, "udp", "[::]", "[::ffff:127.0.0.1]", isEmpty);
			doTestPacket(t, "udp", "[::]", "127.0.0.1", isEmpty);
			doTestPacket(t, "udp", "0.0.0.0", "[::ffff:127.0.0.1]", isEmpty);
		}
	}
}

func TestUnixDatagramServer(t *testing.T) {
	for _, isEmpty := range []bool{false, true} {
		os.Remove("/tmp/gotest1.net");
		os.Remove("/tmp/gotest1.net.local");
		doTestPacket(t, "unixgram", "/tmp/gotest1.net", "/tmp/gotest1.net", isEmpty);
		os.Remove("/tmp/gotest1.net");
		os.Remove("/tmp/gotest1.net.local");
		if syscall.OS == "linux" {
			// Test abstract unix domain socket, a Linux-ism
			doTestPacket(t, "unixgram", "@gotest1/net", "@gotest1/net", isEmpty)
		}
	}
}
