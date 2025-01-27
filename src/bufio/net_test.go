// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package bufio_test

import (
	"bufio"
	"io"
	"net"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// TestCopyUnixpacket tests that we can use bufio when copying
// across a unixpacket socket. This used to fail due to an unnecessary
// empty Write call that was interpreted as an EOF.
func TestCopyUnixpacket(t *testing.T) {
	tmpDir := t.TempDir()
	socket := filepath.Join(tmpDir, "unixsock")

	// Start a unixpacket server.
	addr := &net.UnixAddr{
		Name: socket,
		Net:  "unixpacket",
	}
	server, err := net.ListenUnix("unixpacket", addr)
	if err != nil {
		t.Fatal(err)
	}

	// Start a goroutine for the server to accept one connection
	// and read all the data sent on the connection,
	// reporting the number of bytes read on ch.
	ch := make(chan int, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()

		tot := 0
		defer func() {
			ch <- tot
		}()

		serverConn, err := server.Accept()
		if err != nil {
			t.Error(err)
			return
		}

		buf := make([]byte, 1024)
		for {
			n, err := serverConn.Read(buf)
			tot += n
			if err == io.EOF {
				return
			}
			if err != nil {
				t.Error(err)
				return
			}
		}
	}()

	clientConn, err := net.DialUnix("unixpacket", nil, addr)
	if err != nil {
		// Leaves the server goroutine hanging. Oh well.
		t.Fatal(err)
	}

	defer wg.Wait()
	defer clientConn.Close()

	const data = "data"
	r := bufio.NewReader(strings.NewReader(data))
	n, err := io.Copy(clientConn, r)
	if err != nil {
		t.Fatal(err)
	}

	if n != int64(len(data)) {
		t.Errorf("io.Copy returned %d, want %d", n, len(data))
	}

	clientConn.Close()
	tot := <-ch

	if tot != len(data) {
		t.Errorf("server read %d, want %d", tot, len(data))
	}
}
