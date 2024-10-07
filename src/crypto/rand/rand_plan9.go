// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

// This is a pseudorandom generator that seeds itself by reading from
// /dev/random. The read function always returns the full amount asked for, or
// else it returns an error. The generator is a fast key erasure RNG.

var (
	mu      sync.Mutex
	seeded  sync.Once
	seedErr error
	key     [32]byte
)

func read(b []byte) error {
	seeded.Do(func() {
		t := time.AfterFunc(time.Minute, func() {
			println("crypto/rand: blocked for 60 seconds waiting to read random data from the kernel")
		})
		defer t.Stop()
		entropy, err := os.Open(randomDevice)
		if err != nil {
			seedErr = err
			return
		}
		defer entropy.Close()
		_, seedErr = io.ReadFull(entropy, key[:])
	})
	if seedErr != nil {
		return seedErr
	}

	mu.Lock()
	blockCipher, err := aes.NewCipher(key[:])
	if err != nil {
		mu.Unlock()
		return err
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
	blockCipher.Encrypt(key[:aes.BlockSize], block[:])
	inc()
	blockCipher.Encrypt(key[aes.BlockSize:], block[:])
	inc()
	mu.Unlock()

	for len(b) >= aes.BlockSize {
		blockCipher.Encrypt(b[:aes.BlockSize], block[:])
		inc()
		b = b[aes.BlockSize:]
	}
	if len(b) > 0 {
		blockCipher.Encrypt(block[:], block[:])
		copy(b, block[:])
	}
	return nil
}
