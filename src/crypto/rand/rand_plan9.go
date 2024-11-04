// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"internal/byteorder"
	"internal/chacha8rand"
	"io"
	"os"
	"sync"
)

const randomDevice = "/dev/random"

// This is a pseudorandom generator that seeds itself by reading from
// /dev/random. The read function always returns the full amount asked for, or
// else it returns an error.

var (
	mu      sync.Mutex
	seeded  sync.Once
	seedErr error
	state   chacha8rand.State
)

func read(b []byte) error {
	seeded.Do(func() {
		entropy, err := os.Open(randomDevice)
		if err != nil {
			seedErr = err
			return
		}
		defer entropy.Close()
		var seed [32]byte
		_, err = io.ReadFull(entropy, seed[:])
		if err != nil {
			seedErr = err
			return
		}
		state.Init(seed)
	})
	if seedErr != nil {
		return seedErr
	}

	mu.Lock()
	defer mu.Unlock()

	for len(b) >= 8 {
		if x, ok := state.Next(); ok {
			byteorder.BePutUint64(b, x)
			b = b[8:]
		} else {
			state.Refill()
		}
	}
	for len(b) > 0 {
		if x, ok := state.Next(); ok {
			var buf [8]byte
			byteorder.BePutUint64(buf[:], x)
			n := copy(b, buf[:])
			b = b[n:]
		} else {
			state.Refill()
		}
	}
	state.Reseed()

	return nil
}
