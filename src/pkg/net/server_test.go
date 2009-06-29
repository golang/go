// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io";
	"net";
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
		fd.Write(buf[0:n])
	}
	done <- 1
}

func runServe(t *testing.T, network, addr string, listening chan<- string, done chan<- int) {
	l, err := net.Listen(network, addr);
	if err != nil {
		t.Fatalf("net.Listen(%q, %q) = _, %v", network, addr, err);
	}
	listening <- l.Addr();

	for {
		fd, addr, err := l.Accept();
		if err != nil {
			break;
		}
		echodone := make(chan int);
		go runEcho(fd, echodone);
		<-echodone;	// make sure Echo stops
		l.Close();
	}
	done <- 1
}

func connect(t *testing.T, network, addr string) {
	fd, err := net.Dial(network, "", addr);
	if err != nil {
		t.Fatalf("net.Dial(%q, %q, %q) = _, %v", network, "", addr, err);
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
		dialaddr += addr[strings.LastIndex(addr, ":"):len(addr)];
	}
	connect(t, network, dialaddr);
	<-done;	// make sure server stopped
}

func TestTcpServer(t *testing.T) {
	doTest(t,  "tcp", "0.0.0.0", "127.0.0.1");
	doTest(t, "tcp", "[::]", "[::ffff:127.0.0.1]");
	doTest(t, "tcp", "[::]", "127.0.0.1");
	doTest(t, "tcp", "", "127.0.0.1");
	doTest(t, "tcp", "0.0.0.0", "[::ffff:127.0.0.1]");
}

func TestUnixServer(t *testing.T) {
	doTest(t, "unix", "/tmp/gotest.net", "/tmp/gotest.net");
	if syscall.OS == "linux" {
		// Test abstract unix domain socket, a Linux-ism
		doTest(t, "unix", "@gotest/net", "@gotest/net");
	}
}
