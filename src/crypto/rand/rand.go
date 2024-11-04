// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rand implements a cryptographically secure
// random number generator.
package rand

import (
	"crypto/internal/boring"
	"crypto/internal/sysrand"
	"io"
	_ "unsafe"
)

// Reader is a global, shared instance of a cryptographically
// secure random number generator. It is safe for concurrent use.
//
//   - On Linux, FreeBSD, Dragonfly, and Solaris, Reader uses getrandom(2).
//   - On legacy Linux (< 3.17), Reader opens /dev/urandom on first use.
//   - On macOS, iOS, and OpenBSD Reader, uses arc4random_buf(3).
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

type reader struct{}

func (r *reader) Read(b []byte) (n int, err error) {
	boring.Unreachable()
	sysrand.Read(b)
	return len(b), nil
}

// fatal is [runtime.fatal], pushed via linkname.
//
//go:linkname fatal
func fatal(string)

// Read fills b with cryptographically secure random bytes. It never returns an
// error, and always fills b entirely.
//
// Read calls [io.ReadFull] on [Reader] and crashes the program irrecoverably if
// an error is returned. The default Reader uses operating system APIs that are
// documented to never return an error on all but legacy Linux systems.
func Read(b []byte) (n int, err error) {
	// We don't want b to escape to the heap, but escape analysis can't see
	// through a potentially overridden Reader, so we special-case the default
	// case which we can keep non-escaping, and in the general case we read into
	// a heap buffer and copy from it.
	if r, ok := Reader.(*reader); ok {
		_, err = r.Read(b)
	} else {
		bb := make([]byte, len(b))
		_, err = io.ReadFull(Reader, bb)
		copy(b, bb)
	}
	if err != nil {
		fatal("crypto/rand: failed to read random data (see https://go.dev/issue/66821): " + err.Error())
		panic("unreachable") // To be sure.
	}
	return len(b), nil
}
