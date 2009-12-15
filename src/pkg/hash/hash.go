// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hash

import "io"

// Hash is the common interface implemented by all hash functions.
// The Write method never returns an error.
// Sum returns the bytes of integer hash codes in big-endian order.
type Hash interface {
	io.Writer
	Sum() []byte
	Reset()
	Size() int // number of bytes Sum returns
}

// Hash32 is the common interface implemented by all 32-bit hash functions.
type Hash32 interface {
	Hash
	Sum32() uint32
}
