// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/syscall/unix"
	"testing"
)

func TestMaxAckBacklog(t *testing.T) {
	n := 196602
	major, minor := unix.KernelVersion()
	backlog := maxAckBacklog(n)
	expected := 1<<16 - 1
	if major > 4 || (major == 4 && minor >= 1) {
		expected = n
	}
	if backlog != expected {
		t.Fatalf(`Kernel version: "%d.%d", sk_max_ack_backlog mismatch, got %d, want %d`, major, minor, backlog, expected)
	}
}
