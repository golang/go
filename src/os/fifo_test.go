// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package os_test

import (
	"bufio"
	"bytes"
	"fmt"
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
