// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

// Unix cryptographically secure pseudorandom number
// generator.

package rand

import (
	"errors"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

import "crypto/internal/boring"

const urandomDevice = "/dev/urandom"

func init() {
	if boring.Enabled {
		Reader = boring.RandReader
		return
	}
	Reader = &reader{}
}

// A reader satisfies reads by reading from urandomDevice
type reader struct {
	f    io.Reader
	mu   sync.Mutex
	used uint32 // Atomic: 0 - never used, 1 - used, but f == nil, 2 - used, and f != nil
}

// altGetRandom if non-nil specifies an OS-specific function to get
// urandom-style randomness.
var altGetRandom func([]byte) (err error)

func warnBlocked() {
	println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
}

func (r *reader) Read(b []byte) (n int, err error) {
	boring.Unreachable()
	if atomic.CompareAndSwapUint32(&r.used, 0, 1) {
		// First use of randomness. Start timer to warn about
		// being blocked on entropy not being available.
		t := time.AfterFunc(time.Minute, warnBlocked)
		defer t.Stop()
	}
	if altGetRandom != nil && altGetRandom(b) == nil {
		return len(b), nil
	}
	if atomic.LoadUint32(&r.used) != 2 {
		r.mu.Lock()
		if atomic.LoadUint32(&r.used) != 2 {
			f, err := os.Open(urandomDevice)
			if err != nil {
				r.mu.Unlock()
				return 0, err
			}
			r.f = hideAgainReader{f}
			atomic.StoreUint32(&r.used, 2)
		}
		r.mu.Unlock()
	}
	return io.ReadFull(r.f, b)
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
