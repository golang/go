// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

// Unix cryptographically secure pseudorandom number
// generator.

package rand

import (
	"bufio"
	"errors"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const urandomDevice = "/dev/urandom"

func init() {
	Reader = &reader{}
}

// A reader satisfies reads by reading from urandomDevice
type reader struct {
	f    io.Reader
	mu   sync.Mutex
	used int32 // atomic; whether this reader has been used
}

// altGetRandom if non-nil specifies an OS-specific function to get
// urandom-style randomness.
var altGetRandom func([]byte) (ok bool)

func warnBlocked() {
	println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
}

func (r *reader) Read(b []byte) (n int, err error) {
	if atomic.CompareAndSwapInt32(&r.used, 0, 1) {
		// First use of randomness. Start timer to warn about
		// being blocked on entropy not being available.
		t := time.AfterFunc(time.Minute, warnBlocked)
		defer t.Stop()
	}
	if altGetRandom != nil && altGetRandom(b) {
		return len(b), nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.f == nil {
		f, err := os.Open(urandomDevice)
		if err != nil {
			return 0, err
		}
		r.f = bufio.NewReader(hideAgainReader{f})
	}
	return r.f.Read(b)
}

// hideAgainReader masks EAGAIN reads from /dev/urandom.
// See golang.org/issue/9205
type hideAgainReader struct {
	r io.Reader
}

func (hr hideAgainReader) Read(p []byte) (n int, err error) {
	n, err = hr.r.Read(p)
	if errors.Is(err, syscall.EAGAIN) {
		err = nil
	}
	return
}
