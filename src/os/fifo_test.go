// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || (linux && !android) || netbsd || openbsd

package os_test

import (
	"errors"
	"internal/syscall/unix"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"testing"
)

func TestFifoEOF(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	fifoName := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(fifoName, 0600); err != nil {
		t.Fatal(err)
	}

	// Per https://pubs.opengroup.org/onlinepubs/9699919799/functions/open.html#tag_16_357_03:
	//
	// - “If O_NONBLOCK is clear, an open() for reading-only shall block the
	//   calling thread until a thread opens the file for writing. An open() for
	//   writing-only shall block the calling thread until a thread opens the file
	//   for reading.”
	//
	// In order to unblock both open calls, we open the two ends of the FIFO
	// simultaneously in separate goroutines.

	rc := make(chan *os.File, 1)
	go func() {
		r, err := os.Open(fifoName)
		if err != nil {
			t.Error(err)
		}
		rc <- r
	}()

	w, err := os.OpenFile(fifoName, os.O_WRONLY, 0)
	if err != nil {
		t.Error(err)
	}

	r := <-rc
	if t.Failed() {
		if r != nil {
			r.Close()
		}
		if w != nil {
			w.Close()
		}
		return
	}

	testPipeEOF(t, r, w)
}

// Issue #59545.
func TestNonPollable(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test with tight loops in short mode")
	}

	// We need to open a non-pollable file.
	// This is almost certainly Linux-specific,
	// but if other systems have non-pollable files,
	// we can add them here.
	const nonPollable = "/dev/net/tun"

	f, err := os.OpenFile(nonPollable, os.O_RDWR, 0)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrPermission) || testenv.SyscallIsNotSupported(err) {
			t.Skipf("can't open %q: %v", nonPollable, err)
		}
		t.Fatal(err)
	}
	f.Close()

	// On a Linux laptop, before the problem was fixed,
	// this test failed about 50% of the time with this
	// number of iterations.
	// It takes about 1/2 second when it passes.
	const attempts = 20000

	start := make(chan bool)
	var wg sync.WaitGroup
	wg.Add(1)
	defer wg.Wait()
	go func() {
		defer wg.Done()
		close(start)
		for i := 0; i < attempts; i++ {
			f, err := os.OpenFile(nonPollable, os.O_RDWR, 0)
			if err != nil {
				t.Error(err)
				return
			}
			if err := f.Close(); err != nil {
				t.Error(err)
				return
			}
		}
	}()

	dir := t.TempDir()
	<-start
	for i := 0; i < attempts; i++ {
		name := filepath.Join(dir, strconv.Itoa(i))
		if err := syscall.Mkfifo(name, 0o600); err != nil {
			t.Fatal(err)
		}
		// The problem only occurs if we use O_NONBLOCK here.
		rd, err := os.OpenFile(name, os.O_RDONLY|syscall.O_NONBLOCK, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		wr, err := os.OpenFile(name, os.O_WRONLY|syscall.O_NONBLOCK, 0o600)
		if err != nil {
			t.Fatal(err)
		}
		const msg = "message"
		if _, err := wr.Write([]byte(msg)); err != nil {
			if errors.Is(err, syscall.EAGAIN) || errors.Is(err, syscall.ENOBUFS) {
				t.Logf("ignoring write error %v", err)
				rd.Close()
				wr.Close()
				continue
			}
			t.Fatalf("write to fifo %d failed: %v", i, err)
		}
		if _, err := rd.Read(make([]byte, len(msg))); err != nil {
			if errors.Is(err, syscall.EAGAIN) || errors.Is(err, syscall.ENOBUFS) {
				t.Logf("ignoring read error %v", err)
				rd.Close()
				wr.Close()
				continue
			}
			t.Fatalf("read from fifo %d failed; %v", i, err)
		}
		if err := rd.Close(); err != nil {
			t.Fatal(err)
		}
		if err := wr.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

// Issue 60211.
func TestOpenFileNonBlocking(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Skipf("can't find executable: %v", err)
	}
	f, err := os.OpenFile(exe, os.O_RDONLY|syscall.O_NONBLOCK, 0666)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	nonblock, err := unix.IsNonblock(int(f.Fd()))
	if err != nil {
		t.Fatal(err)
	}
	if !nonblock {
		t.Errorf("file opened with O_NONBLOCK but in blocking mode")
	}
}

func TestNewFileNonBlocking(t *testing.T) {
	var p [2]int
	if err := syscall.Pipe(p[:]); err != nil {
		t.Fatal(err)
	}
	if err := syscall.SetNonblock(p[0], true); err != nil {
		t.Fatal(err)
	}
	f := os.NewFile(uintptr(p[0]), "pipe")
	nonblock, err := unix.IsNonblock(p[0])
	if err != nil {
		t.Fatal(err)
	}
	if !nonblock {
		t.Error("pipe blocking after NewFile")
	}
	fd := f.Fd()
	if fd != uintptr(p[0]) {
		t.Errorf("Fd returned %d, want %d", fd, p[0])
	}
	nonblock, err = unix.IsNonblock(p[0])
	if err != nil {
		t.Fatal(err)
	}
	if !nonblock {
		t.Error("pipe blocking after Fd")
	}
}
