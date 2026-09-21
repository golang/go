// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

// This file contains functions for loading and storing integers in little endian format.
// These can be replaced with unsafe versions if deemed necessary.

type indexer interface {
	int | int8 | int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64
}

// loadLE8 will load from b at index i.
func loadLE8[I indexer](b []byte, i I) byte {
	return b[i]
}

// loadLE32 will load from b at index i.
func loadLE32[I indexer](b []byte, i I) uint32 {
	// Reslicing and then checking a single index compiles to exactly two
	// bounds checks and a single 4-byte load. Slicing to b[i:i+4] would
	// leave the compiler unable to prove the resulting length, costing an
	// additional bounds check per byte.
	b = b[i:]
	_ = b[3] // bounds check hint to compiler; see golang.org/issue/14808
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

// loadLE64 will load from b at index i.
func loadLE64[I indexer](b []byte, i I) uint64 {
	// See loadLE32 for why the load is structured this way.
	b = b[i:]
	_ = b[7] // bounds check hint to compiler; see golang.org/issue/14808
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

// storeLE64 will store v at start of b.
func storeLE64(b []byte, v uint64) {
	_ = b[7] // early bounds check to guarantee safety of writes below
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}
