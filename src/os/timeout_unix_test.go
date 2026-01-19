// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !plan9 && !wasip1 && !windows

package os_test

import (
	"os"
	"os/signal"
	"syscall"
	"testing"
	"time"
)

func init() {
	pipeDeadlinesTestCases = []pipeDeadlineTest{{
		"anonymous pipe",
		func(t *testing.T) (r, w *os.File) {
			r, w, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			return r, w
		},
	}}
}

// Closing a TTY while reading from it should not hang.  Issue 23943.
func TestTTYClose(t *testing.T) {
	// Ignore SIGTTIN in case we are running in the background.
	signal.Ignore(syscall.SIGTTIN)
	defer signal.Reset(syscall.SIGTTIN)

	f, err := os.Open("/dev/tty")
	if err != nil {
		t.Skipf("skipping because opening /dev/tty failed: %v", err)
	}

	go func() {
		var buf [1]byte
		f.Read(buf[:])
	}()

	// Give the goroutine a chance to enter the read.
	// It doesn't matter much if it occasionally fails to do so,
	// we won't be testing what we want to test but the test will pass.
	time.Sleep(time.Millisecond)

	c := make(chan bool)
	go func() {
		defer close(c)
		f.Close()
	}()

	select {
	case <-c:
	case <-time.After(time.Second):
		t.Error("timed out waiting for close")
	}

	// On some systems the goroutines may now be hanging.
	// There's not much we can do about that.
}
