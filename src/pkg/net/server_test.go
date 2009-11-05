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
			break;
		}
		fd.Write(buf[0:n]);
	}
	done <- 1;
}

func runServe(t *testing.T, network, addr string, listening chan<- string, done chan<- int) {
	l, err := Listen(network, addr);
	if err != nil {
		t.Fatalf("net.Listen(%q, %q) = _, %v", network, addr, err);
	}
	listening <- l.Addr().String();

	for {
		fd, err := l.Accept();
		if err != nil {
			break;
		}
		echodone := make(chan int);
		go runEcho(fd, echodone);
		<-echodone;	// make sure Echo stops
		l.Close();
	}
	done <- 1;
}

func connect(t *testing.T, network, addr string) {
	var laddr string;
	if network == "unixgram" {
		laddr = addr+".local";
	}
	fd, err := Dial(network, laddr, addr);
	if err != nil {
		t.Fatalf("net.Dial(%q, %q, %q) = _, %v", network, laddr, addr, err);
	}

	b := strings.Bytes("hello, world\n");
	var b1 [100]byte;

	n, errno := fd.Write(b);
	if n != len(b) {
		t.Fatalf("fd.Write(%q) = %d, %v", b, n, errno);
	}

	n, errno = fd.Read(&b1);
	if n != len(b) {
		t.Fatalf("fd.Read() = %d, %v", n, errno);
	}
	fd.Close();
}

func doTest(t *testing.T, network, listenaddr, dialaddr string) {
	t.Logf("Test %s %s %s\n", network, listenaddr, dialaddr);
	listening := make(chan string);
	done := make(chan int);
	if network == "tcp" {
		listenaddr += ":0";	// any available port
	}
	go runServe(t, network, listenaddr, listening, done);
	addr := <-listening;	// wait for server to start
	if network == "tcp" {
		dialaddr += addr[strings.LastIndex(addr, ":") : len(addr)];
	}
	connect(t, network, dialaddr);
	<-done;	// make sure server stopped
}

func TestTCPServer(t *testing.T) {
	doTest(t, "tcp", "0.0.0.0", "127.0.0.1");
	doTest(t, "tcp", "[::]", "[::ffff:127.0.0.1]");
	doTest(t, "tcp", "[::]", "127.0.0.1");
	doTest(t, "tcp", "", "127.0.0.1");
	doTest(t, "tcp", "0.0.0.0", "[::ffff:127.0.0.1]");
}

func TestUnixServer(t *testing.T) {
	os.Remove("/tmp/gotest.net");
	doTest(t, "unix", "/tmp/gotest.net", "/tmp/gotest.net");
	os.Remove("/tmp/gotest.net");
	if syscall.OS == "linux" {
		// Test abstract unix domain socket, a Linux-ism
		doTest(t, "unix", "@gotest/net", "@gotest/net");
	}
}

func runPacket(t *testing.T, network, addr string, listening chan<- string, done chan<- int) {
	c, err := ListenPacket(network, addr);
	if err != nil {
		t.Fatalf("net.ListenPacket(%q, %q) = _, %v", network, addr, err);
	}
	listening <- c.LocalAddr().String();
	c.SetReadTimeout(10e6);	// 10ms
	var buf [1000]byte;
	for {
		n, addr, err := c.ReadFrom(&buf);
		if err == os.EAGAIN {
			if done <- 1 {
				break;
			}
			continue;
		}
		if err != nil {
			break;
		}
		if _, err = c.WriteTo(buf[0:n], addr); err != nil {
			t.Fatalf("WriteTo %v: %v", addr, err);
		}
	}
	c.Close();
	done <- 1;
}

func doTestPacket(t *testing.T, network, listenaddr, dialaddr string) {
	t.Logf("TestPacket %s %s %s\n", network, listenaddr, dialaddr);
	listening := make(chan string);
	done := make(chan int);
	if network == "udp" {
		listenaddr += ":0";	// any available port
	}
	go runPacket(t, network, listenaddr, listening, done);
	addr := <-listening;	// wait for server to start
	if network == "udp" {
		dialaddr += addr[strings.LastIndex(addr, ":") : len(addr)];
	}
	connect(t, network, dialaddr);
	<-done;	// tell server to stop
	<-done;	// wait for stop
}

func TestUDPServer(t *testing.T) {
	doTestPacket(t, "udp", "0.0.0.0", "127.0.0.1");
	doTestPacket(t, "udp", "[::]", "[::ffff:127.0.0.1]");
	doTestPacket(t, "udp", "[::]", "127.0.0.1");
	doTestPacket(t, "udp", "", "127.0.0.1");
	doTestPacket(t, "udp", "0.0.0.0", "[::ffff:127.0.0.1]");
}

func TestUnixDatagramServer(t *testing.T) {
	os.Remove("/tmp/gotest1.net");
	os.Remove("/tmp/gotest1.net.local");
	doTestPacket(t, "unixgram", "/tmp/gotest1.net", "/tmp/gotest1.net");
	os.Remove("/tmp/gotest1.net");
	os.Remove("/tmp/gotest1.net.local");
	if syscall.OS == "linux" {
		// Test abstract unix domain socket, a Linux-ism
		doTestPacket(t, "unixgram", "@gotest1/net", "@gotest1/net");
	}
}
