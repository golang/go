// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan9 cryptographically secure pseudorandom number
// generator.

package rand

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/binary"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

const randomDevice = "/dev/random"

func init() {
	Reader = &reader{}
}

// reader is a new pseudorandom generator that seeds itself by
// reading from /dev/random. The Read method on the returned
// reader always returns the full amount asked for, or else it
// returns an error. The generator uses the X9.31 algorithm with
// AES-128, reseeding after every 1 MB of generated data.
type reader struct {
	mu                   sync.Mutex
	budget               int // number of bytes that can be generated
	cipher               cipher.Block
	entropy              io.Reader
	entropyUsed          int32 // atomic; whether entropy has been used
	time, seed, dst, key [aes.BlockSize]byte
}

func warnBlocked() {
	println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
}

func (r *reader) readEntropy(b []byte) error {
	if atomic.CompareAndSwapInt32(&r.entropyUsed, 0, 1) {
		// First use of randomness. Start timer to warn about
		// being blocked on entropy not being available.
		t := time.AfterFunc(time.Minute, warnBlocked)
		defer t.Stop()
	}
	var err error
	if r.entropy == nil {
		r.entropy, err = os.Open(randomDevice)
		if err != nil {
			return err
		}
	}
	_, err = io.ReadFull(r.entropy, b)
	return err
}

func (r *reader) Read(b []byte) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	n = len(b)

	for len(b) > 0 {
		if r.budget == 0 {
			err = r.readEntropy(r.seed[0:])
			if err != nil {
				return n - len(b), err
			}
			err = r.readEntropy(r.key[0:])
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
		binary.BigEndian.PutUint64(r.time[:], uint64(ns))
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
