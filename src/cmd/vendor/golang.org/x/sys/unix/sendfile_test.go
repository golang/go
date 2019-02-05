// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin,amd64 darwin,386 dragonfly freebsd linux solaris

package unix_test

import (
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

func TestSendfile(t *testing.T) {
	// Set up source data file.
	tempDir, err := ioutil.TempDir("", "TestSendfile")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)
	name := filepath.Join(tempDir, "source")
	const contents = "contents"
	err = ioutil.WriteFile(name, []byte(contents), 0666)
	if err != nil {
		t.Fatal(err)
	}

	done := make(chan bool)

	// Start server listening on a socket.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("listen failed: %s\n", err)
	}
	defer ln.Close()
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			t.Fatal(err)
		}
		defer conn.Close()
		b, err := ioutil.ReadAll(conn)
		if string(b) != contents {
			t.Errorf("contents not transmitted: got %s (len=%d), want %s", string(b), len(b), contents)
		}
		done <- true
	}()

	// Open source file.
	src, err := os.Open(name)
	if err != nil {
		t.Fatal(err)
	}

	// Send source file to server.
	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	file, err := conn.(*net.TCPConn).File()
	if err != nil {
		t.Fatal(err)
	}
	var off int64
	n, err := unix.Sendfile(int(file.Fd()), int(src.Fd()), &off, len(contents))
	if err != nil {
		t.Errorf("Sendfile failed %s\n", err)
	}
	if n != len(contents) {
		t.Errorf("written count wrong: want %d, got %d", len(contents), n)
	}
	// Note: off is updated on some systems and not others. Oh well.
	// Linux: increments off by the amount sent.
	// Darwin: leaves off unchanged.
	// It would be nice to fix Darwin if we can.
	if off != 0 && off != int64(len(contents)) {
		t.Errorf("offset wrong: god %d, want %d or %d", off, 0, len(contents))
	}
	// The cursor position should be unchanged.
	pos, err := src.Seek(0, 1)
	if err != nil {
		t.Errorf("can't get cursor position %s\n", err)
	}
	if pos != 0 {
		t.Errorf("cursor position wrong: got %d, want 0", pos)
	}

	file.Close() // Note: required to have the close below really send EOF to the server.
	conn.Close()

	// Wait for server to close.
	<-done
}
