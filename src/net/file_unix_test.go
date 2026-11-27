// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package net

import (
	"errors"
	"internal/syscall/unix"
	"os"
	"syscall"
	"testing"
)

// For backward compatibility, opening a net.Conn, turning it into an os.File,
// and calling the Fd method should return a blocking descriptor.
func TestFileFdBlocks(t *testing.T) {
	if !testableNetwork("unix") {
		t.Skipf("skipping: unix sockets not supported")
	}

	ls := newLocalServer(t, "unix")
	defer ls.teardown()

	errc := make(chan error, 1)
	done := make(chan bool)
	handler := func(ls *localServer, ln Listener) {
		server, err := ln.Accept()
		errc <- err
		if err != nil {
			return
		}
		defer server.Close()
		<-done
	}
	if err := ls.buildup(handler); err != nil {
		t.Fatal(err)
	}
	defer close(done)

	client, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	if err := <-errc; err != nil {
		t.Fatalf("server error: %v", err)
	}

	// The socket should be non-blocking.
	rawconn, err := client.(*UnixConn).SyscallConn()
	if err != nil {
		t.Fatal(err)
	}
	err = rawconn.Control(func(fd uintptr) {
		nonblock, err := unix.IsNonblock(int(fd))
		if err != nil {
			t.Fatal(err)
		}
		if !nonblock {
			t.Fatal("unix socket is in blocking mode")
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	file, err := client.(*UnixConn).File()
	if err != nil {
		t.Fatal(err)
	}

	// At this point the descriptor should still be non-blocking.
	rawconn, err = file.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}
	err = rawconn.Control(func(fd uintptr) {
		nonblock, err := unix.IsNonblock(int(fd))
		if err != nil {
			t.Fatal(err)
		}
		if !nonblock {
			t.Fatal("unix socket as os.File is in blocking mode")
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	fd := file.Fd()

	// Calling Fd should have put the descriptor into blocking mode.
	nonblock, err := unix.IsNonblock(int(fd))
	if err != nil {
		t.Fatal(err)
	}
	if nonblock {
		t.Error("unix socket through os.File.Fd is non-blocking")
	}
}

// An SCTP one-to-many style socket (RFC 6458, Section 3.1.1) is an
// AF_INET or AF_INET6 socket of type SOCK_SEQPACKET, which this package
// does not support. FileConn, FileListener and FilePacketConn must
// return an error for it rather than panic. See go.dev/issue/82051.
func TestFileUnsupportedSocketType(t *testing.T) {
	const ipprotoSCTP = 132 // syscall.IPPROTO_SCTP is not defined on all platforms

	tests := []struct {
		name   string
		family int
		sa     syscall.Sockaddr
		ok     bool
	}{
		{"IPv4", syscall.AF_INET, &syscall.SockaddrInet4{Addr: [4]byte{127, 0, 0, 1}}, supportsIPv4()},
		{"IPv6", syscall.AF_INET6, &syscall.SockaddrInet6{Addr: [16]byte{15: 1}}, supportsIPv6()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !tt.ok {
				t.Skipf("skipping: %s not supported", tt.name)
			}
			s, err := syscall.Socket(tt.family, syscall.SOCK_SEQPACKET, ipprotoSCTP)
			if err != nil {
				t.Skipf("skipping: cannot create SCTP socket: %v", err)
			}
			f := os.NewFile(uintptr(s), "sctp")
			defer f.Close()
			if err := syscall.Bind(s, tt.sa); err != nil {
				t.Skipf("skipping: cannot bind SCTP socket: %v", err)
			}
			if err := syscall.Listen(s, 1); err != nil {
				t.Skipf("skipping: cannot listen on SCTP socket: %v", err)
			}

			c, err := FileConn(f)
			if err == nil {
				c.Close()
			}
			if !errors.Is(err, syscall.EPROTONOSUPPORT) {
				t.Errorf("FileConn: got %v; want %v", err, syscall.EPROTONOSUPPORT)
			}
			ln, err := FileListener(f)
			if err == nil {
				ln.Close()
			}
			if !errors.Is(err, syscall.EPROTONOSUPPORT) {
				t.Errorf("FileListener: got %v; want %v", err, syscall.EPROTONOSUPPORT)
			}
			pc, err := FilePacketConn(f)
			if err == nil {
				pc.Close()
			}
			if !errors.Is(err, syscall.EPROTONOSUPPORT) {
				t.Errorf("FilePacketConn: got %v; want %v", err, syscall.EPROTONOSUPPORT)
			}
		})
	}
}
