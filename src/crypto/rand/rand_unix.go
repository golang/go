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
	used bool // whether this reader has been used
}

// altGetRandom if non-nil specifies an OS-specific function to get
// urandom-style randomness.
var altGetRandom func([]byte) (ok bool)

// batched returns a function that calls f to populate a []byte by chunking it
// into subslices of, at most, readMax bytes, buffering min(readMax, 4096)
// bytes at a time.
func batched(f func([]byte) error, readMax int) func([]byte) bool {
	bufferSize := 4096
	if bufferSize > readMax {
		bufferSize = readMax
	}
	fullBuffer := make([]byte, bufferSize)
	var buf []byte
	return func(out []byte) bool {
		// First we copy any amount remaining in the buffer.
		n := copy(out, buf)
		out, buf = out[n:], buf[n:]

		// Then, if we're requesting more than the buffer size,
		// generate directly into the output, chunked by readMax.
		for len(out) >= len(fullBuffer) {
			read := len(out) - (len(out) % len(fullBuffer))
			if read > readMax {
				read = readMax
			}
			if f(out[:read]) != nil {
				return false
			}
			out = out[read:]
		}

		// If there's a partial block left over, fill the buffer,
		// and copy in the remainder.
		if len(out) > 0 {
			if f(fullBuffer[:]) != nil {
				return false
			}
			buf = fullBuffer[:]
			n = copy(out, buf)
			out, buf = out[n:], buf[n:]
		}

		if len(out) > 0 {
			panic("crypto/rand batching failed to fill buffer")
		}

		return true
	}
}

func warnBlocked() {
	println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
}

func (r *reader) Read(b []byte) (n int, err error) {
	boring.Unreachable()
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.used {
		r.used = true
		// First use of randomness. Start timer to warn about
		// being blocked on entropy not being available.
		t := time.AfterFunc(time.Minute, warnBlocked)
		defer t.Stop()
	}
	if altGetRandom != nil && altGetRandom(b) {
		return len(b), nil
	}
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
