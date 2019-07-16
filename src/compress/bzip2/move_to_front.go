// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bzip2

// moveToFrontDecoder implements a move-to-front list. Such a list is an
// efficient way to transform a string with repeating elements into one with
// many small valued numbers, which is suitable for entropy encoding. It works
// by starting with an initial list of symbols and references symbols by their
// index into that list. When a symbol is referenced, it's moved to the front
// of the list. Thus, a repeated symbol ends up being encoded with many zeros,
// as the symbol will be at the front of the list after the first access.
type moveToFrontDecoder []byte

// newMTFDecoder creates a move-to-front decoder with an explicit initial list
// of symbols.
func newMTFDecoder(symbols []byte) moveToFrontDecoder {
	if len(symbols) > 256 {
		panic("too many symbols")
	}
	return moveToFrontDecoder(symbols)
}

// newMTFDecoderWithRange creates a move-to-front decoder with an initial
// symbol list of 0...n-1.
func newMTFDecoderWithRange(n int) moveToFrontDecoder {
	if n > 256 {
		panic("newMTFDecoderWithRange: cannot have > 256 symbols")
	}

	m := make([]byte, n)
	for i := 0; i < n; i++ {
		m[i] = byte(i)
	}
	return moveToFrontDecoder(m)
}

func (m moveToFrontDecoder) Decode(n int) (b byte) {
	// Implement move-to-front with a simple copy. This approach
	// beats more sophisticated approaches in benchmarking, probably
	// because it has high locality of reference inside of a
	// single cache line (most move-to-front operations have n < 64).
	b = m[n]
	copy(m[1:], m[:n])
	m[0] = b
	return
}

// First returns the symbol at the front of the list.
func (m moveToFrontDecoder) First() byte {
	return m[0]
}
