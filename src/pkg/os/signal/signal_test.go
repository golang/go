// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package signal

import (
	"os"
	"syscall"
	"testing"
	"time"
)

const sighup = syscall.SIGHUP

func waitSig(t *testing.T, c <-chan os.Signal, sig os.Signal) {
	select {
	case s := <-c:
		if s != sig {
			t.Fatalf("signal was %v, want %v", s, sig)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timeout waiting for %v", sig)
	}
}

func TestSignal(t *testing.T) {
	// Ask for SIGHUP
	c := make(chan os.Signal, 1)
	Notify(c, sighup)

	t.Logf("sighup...")
	// Send this process a SIGHUP
	syscall.Kill(syscall.Getpid(), sighup)
	waitSig(t, c, sighup)

	// Ask for everything we can get.
	c1 := make(chan os.Signal, 1)
	Notify(c1)

	t.Logf("sigwinch...")
	// Send this process a SIGWINCH
	syscall.Kill(syscall.Getpid(), syscall.SIGWINCH)
	waitSig(t, c1, syscall.SIGWINCH)

	// Send two more SIGHUPs, to make sure that
	// they get delivered on c1 and that not reading
	// from c does not block everything.
	t.Logf("sigwinch...")
	syscall.Kill(syscall.Getpid(), syscall.SIGHUP)
	waitSig(t, c1, syscall.SIGHUP)
	t.Logf("sigwinch...")
	syscall.Kill(syscall.Getpid(), syscall.SIGHUP)
	waitSig(t, c1, syscall.SIGHUP)

	// The first SIGHUP should be waiting for us on c.
	waitSig(t, c, syscall.SIGHUP)
}
