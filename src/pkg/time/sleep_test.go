// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"os"
	"syscall"
	"testing"
	. "time"
)

func TestSleep(t *testing.T) {
	const delay = int64(100e6)
	go func() {
		Sleep(delay / 2)
		syscall.Kill(os.Getpid(), syscall.SIGCHLD)
	}()
	start := Nanoseconds()
	Sleep(delay)
	duration := Nanoseconds() - start
	if duration < delay {
		t.Fatalf("Sleep(%d) slept for only %d ns", delay, duration)
	}
}

func TestAfter(t *testing.T) {
	const delay = int64(100e6)
	start := Nanoseconds()
	end := <-After(delay)
	if duration := Nanoseconds() - start; duration < delay {
		t.Fatalf("After(%d) slept for only %d ns", delay, duration)
	}
	if min := start + delay; end < min {
		t.Fatalf("After(%d) expect >= %d, got %d", delay, min, end)
	}
}
