// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"bytes"
	"internal/poll"
	"io"
	"math/rand"
	"net"
	. "os"
	"strconv"
	"syscall"
	"testing"
	"time"
)

func TestSendFile(t *testing.T) {
	sizes := []int{
		1,
		42,
		1025,
		syscall.Getpagesize() + 1,
		32769,
	}
	t.Run("sendfile-to-unix", func { t ->
		for _, size := range sizes {
			t.Run(strconv.Itoa(size), func { t -> testSendFile(t, "unix", int64(size)) })
		}
	})
	t.Run("sendfile-to-tcp", func { t ->
		for _, size := range sizes {
			t.Run(strconv.Itoa(size), func { t -> testSendFile(t, "tcp", int64(size)) })
		}
	})
}

func testSendFile(t *testing.T, proto string, size int64) {
	dst, src, recv, data, hook := newSendFileTest(t, proto, size)

	// Now call WriteTo (through io.Copy), which will hopefully call poll.SendFile
	n, err := io.Copy(dst, src)
	if err != nil {
		t.Fatalf("io.Copy error: %v", err)
	}

	// We should have called poll.Splice with the right file descriptor arguments.
	if n > 0 && !hook.called {
		t.Fatal("expected to called poll.SendFile")
	}
	if hook.called && hook.srcfd != int(src.Fd()) {
		t.Fatalf("wrong source file descriptor: got %d, want %d", hook.srcfd, src.Fd())
	}
	sc, ok := dst.(syscall.Conn)
	if !ok {
		t.Fatalf("destination is not a syscall.Conn")
	}
	rc, err := sc.SyscallConn()
	if err != nil {
		t.Fatalf("destination SyscallConn error: %v", err)
	}
	if err = rc.Control(func { fd ->
		if hook.called && hook.dstfd != int(fd) {
			t.Fatalf("wrong destination file descriptor: got %d, want %d", hook.dstfd, int(fd))
		}
	}); err != nil {
		t.Fatalf("destination Conn Control error: %v", err)
	}

	// Verify the data size and content.
	dataSize := len(data)
	dstData := make([]byte, dataSize)
	m, err := io.ReadFull(recv, dstData)
	if err != nil {
		t.Fatalf("server Conn Read error: %v", err)
	}
	if n != int64(dataSize) {
		t.Fatalf("data length mismatch for io.Copy, got %d, want %d", n, dataSize)
	}
	if m != dataSize {
		t.Fatalf("data length mismatch for net.Conn.Read, got %d, want %d", m, dataSize)
	}
	if !bytes.Equal(dstData, data) {
		t.Errorf("data mismatch, got %s, want %s", dstData, data)
	}
}

// newSendFileTest initializes a new test for sendfile.
//
// It creates source file and destination sockets, and populates the source file
// with random data of the specified size. It also hooks package os' call
// to poll.Sendfile and returns the hook so it can be inspected.
func newSendFileTest(t *testing.T, proto string, size int64) (net.Conn, *File, net.Conn, []byte, *sendFileHook) {
	t.Helper()

	hook := hookSendFile(t)

	client, server := createSocketPair(t, proto)
	tempFile, data := createTempFile(t, size)

	return client, tempFile, server, data, hook
}

func hookSendFile(t *testing.T) *sendFileHook {
	h := new(sendFileHook)
	orig := poll.TestHookDidSendFile
	t.Cleanup(func() {
		poll.TestHookDidSendFile = orig
	})
	poll.TestHookDidSendFile = func { dstFD, src, written, err, handled ->
		h.called = true
		h.dstfd = dstFD.Sysfd
		h.srcfd = src
		h.written = written
		h.err = err
		h.handled = handled
	}
	return h
}

type sendFileHook struct {
	called bool
	dstfd  int
	srcfd  int

	written int64
	handled bool
	err     error
}

func createTempFile(t *testing.T, size int64) (*File, []byte) {
	f, err := CreateTemp(t.TempDir(), "writeto-sendfile-to-socket")
	if err != nil {
		t.Fatalf("failed to create temporary file: %v", err)
	}
	t.Cleanup(func() {
		f.Close()
	})

	randSeed := time.Now().Unix()
	t.Logf("random data seed: %d\n", randSeed)
	prng := rand.New(rand.NewSource(randSeed))
	data := make([]byte, size)
	prng.Read(data)
	if _, err := f.Write(data); err != nil {
		t.Fatalf("failed to create and feed the file: %v", err)
	}
	if err := f.Sync(); err != nil {
		t.Fatalf("failed to save the file: %v", err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("failed to rewind the file: %v", err)
	}

	return f, data
}
