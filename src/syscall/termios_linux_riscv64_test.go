// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && riscv64

package syscall

import (
	"testing"
	"unsafe"
)

// TestTermiosLayout verifies that Termios matches the Linux riscv64
// kernel/glibc ABI. A mismatch here causes a C tcgetattr call to write past
// the end of a Go Termios object, corrupting adjacent heap objects.
func TestTermiosLayout(t *testing.T) {
	if got := unsafe.Sizeof(Termios{}); got != 60 {
		t.Fatalf("sizeof(Termios) = %d, want 60", got)
	}
	if got := unsafe.Offsetof(Termios{}.Line); got != 16 {
		t.Fatalf("offsetof(Termios.Line) = %d, want 16", got)
	}
	if got := unsafe.Offsetof(Termios{}.Cc); got != 17 {
		t.Fatalf("offsetof(Termios.Cc) = %d, want 17", got)
	}
	if got := unsafe.Offsetof(Termios{}.Ispeed); got != 52 {
		t.Fatalf("offsetof(Termios.Ispeed) = %d, want 52", got)
	}
	if got := unsafe.Offsetof(Termios{}.Ospeed); got != 56 {
		t.Fatalf("offsetof(Termios.Ospeed) = %d, want 56", got)
	}
}
