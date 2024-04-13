// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

// Unix cryptographically secure pseudorandom number
// generator.

package rand

import (
	"crypto/internal/boring"
	"crypto/rand/internal/getrand"
	"errors"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const urandomDevice = "/dev/urandom"

var randReader = &reader{}

func init() {
	if boring.Enabled {
		Reader = boring.RandReader
	}
}

// A reader satisfies reads by reading from urandomDevice
type reader struct {
	f    hideAgainFileReader
	mu   sync.Mutex
	used atomic.Uint32 // Atomic: 0 - never used, 1 - used, but f == nil, 2 - used, and f != nil
}

func warnBlocked() {
	println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
}

func (r *reader) Read(b []byte) (n int, err error) {
	if boring.Enabled {
		return boring.RandReader.Read(b)
	}
	boring.Unreachable()

	if r.used.CompareAndSwap(0, 1) {
		// First use of randomness. Start timer to warn about
		// being blocked on entropy not being available.
		t := time.AfterFunc(time.Minute, warnBlocked)
		defer t.Stop()
	}

	if getrand.GetRandom(b) == nil {
		return len(b), nil
	}

	if r.used.Load() != 2 {
		r.mu.Lock()
		if r.used.Load() != 2 {
			f, err := os.Open(urandomDevice)
			if err != nil {
				r.mu.Unlock()
				return 0, err
			}
			r.f = hideAgainFileReader{f}
			r.used.Store(2)
		}
		r.mu.Unlock()
	}

	return r.f.ReadFull(b)
}

// hideAgainFileReader masks EAGAIN reads from /dev/urandom.
// See golang.org/issue/9205
type hideAgainFileReader struct {
	f *os.File
}

func (hr hideAgainFileReader) Read(p []byte) (n int, err error) {
	n, err = hr.f.Read(p)
	if errors.Is(err, syscall.EAGAIN) {
		err = nil
	}
	return
}

func (hr hideAgainFileReader) ReadFull(p []byte) (n int, err error) {
	for n < len(p) && err == nil {
		var nn int
		nn, err = hr.Read(p[n:])
		n += nn
	}
	if n >= len(p) {
		err = nil
	} else if n > 0 && err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return
}
