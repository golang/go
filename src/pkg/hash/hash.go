// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hash provides interfaces for hash functions.
package hash

import "io"

// Hash is the common interface implemented by all hash functions.
type Hash interface {
	// Write adds more data to the running hash.
	// It never returns an error.
	io.Writer

	// Sum appends the current hash in the same manner as append(), without
	// changing the underlying hash state.
	Sum(in []byte) []byte

	// Reset resets the hash to one with zero bytes written.
	Reset()

	// Size returns the number of bytes Sum will return.
	Size() int
}

// Hash32 is the common interface implemented by all 32-bit hash functions.
type Hash32 interface {
	Hash
	Sum32() uint32
}

// Hash64 is the common interface implemented by all 64-bit hash functions.
type Hash64 interface {
	Hash
	Sum64() uint64
}
