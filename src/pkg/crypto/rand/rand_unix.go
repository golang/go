// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd plan9

// Unix cryptographically secure pseudorandom number
// generator.

package rand

import (
	"bufio"
	"crypto/aes"
	"crypto/cipher"
	"io"
	"os"
	"runtime"
	"sync"
	"time"
)

// Easy implementation: read from /dev/urandom.
// This is sufficient on Linux, OS X, and FreeBSD.

func init() {
	if runtime.GOOS == "plan9" {
		Reader = newReader(nil)
	} else {
		Reader = &devReader{name: "/dev/urandom"}
	}
}

// A devReader satisfies reads by reading the file named name.
type devReader struct {
	name string
	f    io.Reader
	mu   sync.Mutex
}

func (r *devReader) Read(b []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.f == nil {
		f, err := os.Open(r.name)
		if f == nil {
			return 0, err
		}
		if runtime.GOOS == "plan9" {
			r.f = f
		} else {
			r.f = bufio.NewReader(f)
		}
	}
	return r.f.Read(b)
}

// Alternate pseudo-random implementation for use on
// systems without a reliable /dev/urandom.

// newReader returns a new pseudorandom generator that
// seeds itself by reading from entropy.  If entropy == nil,
// the generator seeds itself by reading from the system's
// random number generator, typically /dev/random.
// The Read method on the returned reader always returns
// the full amount asked for, or else it returns an error.
//
// The generator uses the X9.31 algorithm with AES-128,
// reseeding after every 1 MB of generated data.
func newReader(entropy io.Reader) io.Reader {
	if entropy == nil {
		entropy = &devReader{name: "/dev/random"}
	}
	return &reader{entropy: entropy}
}

type reader struct {
	mu                   sync.Mutex
	budget               int // number of bytes that can be generated
	cipher               cipher.Block
	entropy              io.Reader
	time, seed, dst, key [aes.BlockSize]byte
}

func (r *reader) Read(b []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	n = len(b)

	for len(b) > 0 {
		if r.budget == 0 {
			_, err := io.ReadFull(r.entropy, r.seed[0:])
			if err != nil {
				return n - len(b), err
			}
			_, err = io.ReadFull(r.entropy, r.key[0:])
			if err != nil {
				return n - len(b), err
			}
			r.cipher, err = aes.NewCipher(r.key[0:])
			if err != nil {
				return n - len(b), err
			}
			r.budget = 1 << 20 // reseed after generating 1MB
		}
		r.budget -= aes.BlockSize

		// ANSI X9.31 (== X9.17) algorithm, but using AES in place of 3DES.
		//
		// single block:
		// t = encrypt(time)
		// dst = encrypt(t^seed)
		// seed = encrypt(t^dst)
		ns := time.Now().UnixNano()
		r.time[0] = byte(ns >> 56)
		r.time[1] = byte(ns >> 48)
		r.time[2] = byte(ns >> 40)
		r.time[3] = byte(ns >> 32)
		r.time[4] = byte(ns >> 24)
		r.time[5] = byte(ns >> 16)
		r.time[6] = byte(ns >> 8)
		r.time[7] = byte(ns)
		r.cipher.Encrypt(r.time[0:], r.time[0:])
		for i := 0; i < aes.BlockSize; i++ {
			r.dst[i] = r.time[i] ^ r.seed[i]
		}
		r.cipher.Encrypt(r.dst[0:], r.dst[0:])
		for i := 0; i < aes.BlockSize; i++ {
			r.seed[i] = r.time[i] ^ r.dst[i]
		}
		r.cipher.Encrypt(r.seed[0:], r.seed[0:])

		m := copy(b, r.dst[0:])
		b = b[m:]
	}

	return n, nil
}
