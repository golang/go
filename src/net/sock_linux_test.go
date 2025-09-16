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
	backlog := maxAckBacklog(n)
	expected := 1<<16 - 1
	if unix.KernelVersionGE(4, 1) {
		expected = n
	}
	if backlog != expected {
		t.Fatalf(`sk_max_ack_backlog mismatch, got %d, want %d`, backlog, expected)
	}
}
