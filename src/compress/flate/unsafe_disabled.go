// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"internal/byteorder"
)

type indexer interface {
	int | int8 | int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64
}

// loadLE8 will load from b at index i.
func loadLE8[I indexer](b []byte, i I) byte {
	return b[i]
}

// loadLE32 will load from b at index i.
func loadLE32[I indexer](b []byte, i I) uint32 {
	return byteorder.LEUint32(b[i:])
}

// loadLE64 will load from b at index i.
func loadLE64[I indexer](b []byte, i I) uint64 {
	return byteorder.LEUint64(b[i:])
}

// storeLE64 will store v at start of b.
func storeLE64(b []byte, v uint64) {
	byteorder.LEPutUint64(b, v)
}
