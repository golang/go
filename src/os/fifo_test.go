// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || (linux && !android) || netbsd || openbsd

package os_test

import (
	"os"
	"path/filepath"
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
