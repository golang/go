// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan9 cryptographically secure pseudorandom number
// generator.

package rand

import (
	"crypto/aes"
	"internal/byteorder"
	"io"
	"os"
	"sync"
	"time"
)

const randomDevice = "/dev/random"

var randReader = &reader{}

// reader is a new pseudorandom generator that seeds itself by
// reading from /dev/random. The Read method on the returned
// reader always returns the full amount asked for, or else it
// returns an error. The generator is a fast key erasure RNG.
type reader struct {
	mu      sync.Mutex
	seeded  sync.Once
	seedErr error
	key     [32]byte
}

func (r *reader) Read(b []byte) (n int, err error) {
	r.seeded.Do(func() {
		t := time.AfterFunc(time.Minute, func() {
			println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
		})
		defer t.Stop()
		entropy, err := os.Open(randomDevice)
		if err != nil {
			r.seedErr = err
			return
		}
		defer entropy.Close()
		_, r.seedErr = io.ReadFull(entropy, r.key[:])
	})
	if r.seedErr != nil {
		return 0, r.seedErr
	}

	r.mu.Lock()
	blockCipher, err := aes.NewCipher(r.key[:])
	if err != nil {
		r.mu.Unlock()
		return 0, err
	}
	var (
		counter uint64
		block   [aes.BlockSize]byte
	)
	inc := func() {
		counter++
		if counter == 0 {
			panic("crypto/rand counter wrapped")
		}
		byteorder.LePutUint64(block[:], counter)
	}
	blockCipher.Encrypt(r.key[:aes.BlockSize], block[:])
	inc()
	blockCipher.Encrypt(r.key[aes.BlockSize:], block[:])
	inc()
	r.mu.Unlock()

	n = len(b)
	for len(b) >= aes.BlockSize {
		blockCipher.Encrypt(b[:aes.BlockSize], block[:])
		inc()
		b = b[aes.BlockSize:]
	}
	if len(b) > 0 {
		blockCipher.Encrypt(block[:], block[:])
		copy(b, block[:])
	}
	return n, nil
}
