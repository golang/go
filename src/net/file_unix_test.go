// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package net

import (
	"internal/syscall/unix"
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
