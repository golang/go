// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package net

import (
	"internal/testpty"
	"io"
	"os"
	"sync"
	"syscall"
	"testing"
)

// Issue 70763: test that we don't fail on sendfile from a tty.
func TestCopyFromTTY(t *testing.T) {
	pty, ttyName, err := testpty.Open()
	if err != nil {
		t.Skipf("skipping test because pty open failed: %v", err)
	}
	defer pty.Close()

	// Use syscall.Open so that the tty is blocking.
	ttyFD, err := syscall.Open(ttyName, syscall.O_RDWR, 0)
	if err != nil {
		t.Skipf("skipping test because tty open failed: %v", err)
	}
	defer syscall.Close(ttyFD)

	tty := os.NewFile(uintptr(ttyFD), "tty")
	defer tty.Close()

	ln := newLocalListener(t, "tcp")
	defer ln.Close()

	ch := make(chan bool)

	const data = "data\n"

	var wg sync.WaitGroup
	defer wg.Wait()

	wg.Add(1)
	go func() {
		defer wg.Done()
		conn, err := ln.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		defer conn.Close()

		buf := make([]byte, len(data))
		if _, err := io.ReadFull(conn, buf); err != nil {
			t.Error(err)
		}

		ch <- true
	}()

	conn, err := Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	wg.Add(1)
	go func() {
		defer wg.Done()
		if _, err := pty.Write([]byte(data)); err != nil {
			t.Error(err)
		}
		<-ch
		if err := pty.Close(); err != nil {
			t.Error(err)
		}
	}()

	lr := io.LimitReader(tty, int64(len(data)))
	if _, err := io.Copy(conn, lr); err != nil {
		t.Error(err)
	}
}
