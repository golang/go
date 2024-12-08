// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rand provides cryptographically secure random bytes from the
// operating system.
package sysrand

import (
	"os"
	"sync"
	"sync/atomic"
	"time"
	_ "unsafe"
)

var firstUse atomic.Bool

func warnBlocked() {
	println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
}

// fatal is [runtime.fatal], pushed via linkname.
//
//go:linkname fatal
func fatal(string)

var testingOnlyFailRead bool

// Read fills b with cryptographically secure random bytes from the operating
// system. It always fills b entirely and crashes the program irrecoverably if
// an error is encountered. The operating system APIs are documented to never
// return an error on all but legacy Linux systems.
func Read(b []byte) {
	if firstUse.CompareAndSwap(false, true) {
		// First use of randomness. Start timer to warn about
		// being blocked on entropy not being available.
		t := time.AfterFunc(time.Minute, warnBlocked)
		defer t.Stop()
	}
	if err := read(b); err != nil || testingOnlyFailRead {
		var errStr string
		if !testingOnlyFailRead {
			errStr = err.Error()
		} else {
			errStr = "testing simulated failure"
		}
		fatal("crypto/rand: failed to read random data (see https://go.dev/issue/66821): " + errStr)
		panic("unreachable") // To be sure.
	}
}

// The urandom fallback is only used on Linux kernels before 3.17 and on AIX.

var urandomOnce sync.Once
var urandomFile *os.File
var urandomErr error

func urandomRead(b []byte) error {
	urandomOnce.Do(func() {
		urandomFile, urandomErr = os.Open("/dev/urandom")
	})
	if urandomErr != nil {
		return urandomErr
	}
	for len(b) > 0 {
		n, err := urandomFile.Read(b)
		// Note that we don't ignore EAGAIN because it should not be possible to
		// hit for a blocking read from urandom, although there were
		// unreproducible reports of it at https://go.dev/issue/9205.
		if err != nil {
			return err
		}
		b = b[n:]
	}
	return nil
}
