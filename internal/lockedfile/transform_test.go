// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// js does not support inter-process file locking.
//
//go:build !js
// +build !js

package lockedfile_test

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/tools/internal/lockedfile"
)

func isPowerOf2(x int) bool {
	return x > 0 && x&(x-1) == 0
}

func roundDownToPowerOf2(x int) int {
	if x <= 0 {
		panic("nonpositive x")
	}
	bit := 1
	for x != bit {
		x = x &^ bit
		bit <<= 1
	}
	return x
}

func TestTransform(t *testing.T) {
	dir, remove := mustTempDir(t)
	defer remove()
	path := filepath.Join(dir, "blob.bin")

	const maxChunkWords = 8 << 10
	buf := make([]byte, 2*maxChunkWords*8)
	for i := uint64(0); i < 2*maxChunkWords; i++ {
		binary.LittleEndian.PutUint64(buf[i*8:], i)
	}
	if err := lockedfile.Write(path, bytes.NewReader(buf[:8]), 0666); err != nil {
		t.Fatal(err)
	}

	var attempts int64 = 128
	if !testing.Short() {
		attempts *= 16
	}
	const parallel = 32

	var sem = make(chan bool, parallel)

	for n := attempts; n > 0; n-- {
		sem <- true
		go func() {
			defer func() { <-sem }()

			time.Sleep(time.Duration(rand.Intn(100)) * time.Microsecond)
			chunkWords := roundDownToPowerOf2(rand.Intn(maxChunkWords) + 1)
			offset := rand.Intn(chunkWords)

			err := lockedfile.Transform(path, func(data []byte) (chunk []byte, err error) {
				chunk = buf[offset*8 : (offset+chunkWords)*8]

				if len(data)&^7 != len(data) {
					t.Errorf("read %d bytes, but each write is an integer multiple of 8 bytes", len(data))
					return chunk, nil
				}

				words := len(data) / 8
				if !isPowerOf2(words) {
					t.Errorf("read %d 8-byte words, but each write is a power-of-2 number of words", words)
					return chunk, nil
				}

				u := binary.LittleEndian.Uint64(data)
				for i := 1; i < words; i++ {
					next := binary.LittleEndian.Uint64(data[i*8:])
					if next != u+1 {
						t.Errorf("wrote sequential integers, but read integer out of sequence at offset %d", i)
						return chunk, nil
					}
					u = next
				}

				return chunk, nil
			})

			if err != nil {
				t.Errorf("unexpected error from Transform: %v", err)
			}
		}()
	}

	for n := parallel; n > 0; n-- {
		sem <- true
	}
}
