// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package signal

import (
	"os"
	"syscall"
	"testing"
	"time"
)

const prSetKeepCaps = 8

// This test validates that syscall.AllThreadsSyscall() can reliably
// reach all 'm' (threads) of the nocgo runtime even when one thread
// is blocked waiting to receive signals from the kernel. This monitors
// for a regression vs. the fix for #43149.
func TestAllThreadsSyscallSignals(t *testing.T) {
	if _, _, err := syscall.AllThreadsSyscall(syscall.SYS_PRCTL, prSetKeepCaps, 0, 0); err == syscall.ENOTSUP {
		t.Skip("AllThreadsSyscall disabled with cgo")
	}

	sig := make(chan os.Signal, 1)
	Notify(sig, os.Interrupt)

	for i := 0; i <= 100; i++ {
		if _, _, errno := syscall.AllThreadsSyscall(syscall.SYS_PRCTL, prSetKeepCaps, uintptr(i&1), 0); errno != 0 {
			t.Fatalf("[%d] failed to set KEEP_CAPS=%d: %v", i, i&1, errno)
		}
	}

	select {
	case <-time.After(10 * time.Millisecond):
	case <-sig:
		t.Fatal("unexpected signal")
	}
	Stop(sig)
}
