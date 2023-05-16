// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package os_test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/syscall/unix"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"syscall"
	"testing"
	"time"
)

// Issue 24164.
func TestFifoEOF(t *testing.T) {
	switch runtime.GOOS {
	case "android":
		t.Skip("skipping on Android; mkfifo syscall not available")
	}

	dir := t.TempDir()
	fifoName := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(fifoName, 0600); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()

		w, err := os.OpenFile(fifoName, os.O_WRONLY, 0)
		if err != nil {
			t.Error(err)
			return
		}

		defer func() {
			if err := w.Close(); err != nil {
				t.Errorf("error closing writer: %v", err)
			}
		}()

		for i := 0; i < 3; i++ {
			time.Sleep(10 * time.Millisecond)
			_, err := fmt.Fprintf(w, "line %d\n", i)
			if err != nil {
				t.Errorf("error writing to fifo: %v", err)
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}()

	defer wg.Wait()

	r, err := os.Open(fifoName)
	if err != nil {
		t.Fatal(err)
	}

	done := make(chan bool)
	go func() {
		defer close(done)

		defer func() {
			if err := r.Close(); err != nil {
				t.Errorf("error closing reader: %v", err)
			}
		}()

		rbuf := bufio.NewReader(r)
		for {
			b, err := rbuf.ReadBytes('\n')
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Error(err)
				return
			}
			t.Logf("%s\n", bytes.TrimSpace(b))
		}
	}()

	select {
	case <-done:
		// Test succeeded.
	case <-time.After(time.Second):
		t.Error("timed out waiting for read")
		// Close the reader to force the read to complete.
		r.Close()
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
