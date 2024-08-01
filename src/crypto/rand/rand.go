// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rand implements a cryptographically secure
// random number generator.
package rand

import (
	"crypto/internal/boring"
	"io"
	"sync/atomic"
	"time"
)

// Reader is a global, shared instance of a cryptographically
// secure random number generator. It is safe for concurrent use.
//
//   - On Linux, FreeBSD, Dragonfly, and Solaris, Reader uses getrandom(2).
//   - On macOS and iOS, Reader uses arc4random_buf(3).
//   - On OpenBSD, Reader uses getentropy(2).
//   - On NetBSD, Reader uses the kern.arandom sysctl.
//   - On Windows, Reader uses the ProcessPrng API.
//   - On js/wasm, Reader uses the Web Crypto API.
//   - On wasip1/wasm, Reader uses random_get.
var Reader io.Reader

func init() {
	if boring.Enabled {
		Reader = boring.RandReader
		return
	}
	Reader = &reader{}
}

var firstUse atomic.Bool

func warnBlocked() {
	println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
}

type reader struct{}

func (r *reader) Read(b []byte) (n int, err error) {
	boring.Unreachable()
	if firstUse.CompareAndSwap(false, true) {
		// First use of randomness. Start timer to warn about
		// being blocked on entropy not being available.
		t := time.AfterFunc(time.Minute, warnBlocked)
		defer t.Stop()
	}
	if err := read(b); err != nil {
		return 0, err
	}
	return len(b), nil
}

// Read is a helper function that calls Reader.Read using io.ReadFull.
// On return, n == len(b) if and only if err == nil.
func Read(b []byte) (n int, err error) {
	return io.ReadFull(Reader, b)
}

// batched returns a function that calls f to populate a []byte by chunking it
// into subslices of, at most, readMax bytes.
func batched(f func([]byte) error, readMax int) func([]byte) error {
	return func(out []byte) error {
		for len(out) > 0 {
			read := len(out)
			if read > readMax {
				read = readMax
			}
			if err := f(out[:read]); err != nil {
				return err
			}
			out = out[read:]
		}
		return nil
	}
}
