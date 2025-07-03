// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"errors"
)

// maxVarintLenN is the maximum length of a varint-encoded N-bit integer.
const maxVarintLen64 = 10

var (
	errOverflow = errors.New("binary: varint overflows a 64-bit integer")
	errEOB      = errors.New("binary: end of buffer")
)

// TODO deduplicate this function.
func readUvarint(b []byte) (uint64, int, error) {
	var x uint64
	var s uint
	var byt byte
	for i := 0; i < maxVarintLen64 && i < len(b); i++ {
		byt = b[i]
		if byt < 0x80 {
			if i == maxVarintLen64-1 && byt > 1 {
				return x, i, errOverflow
			}
			return x | uint64(byt)<<s, i + 1, nil
		}
		x |= uint64(byt&0x7f) << s
		s += 7
	}
	return x, len(b), errOverflow
}

// putUvarint encodes a uint64 into buf and returns the number of bytes written.
// If the buffer is too small, PutUvarint will panic.
// TODO deduplicate this function.
func putUvarint(buf []byte, x uint64) int {
	i := 0
	for x >= 0x80 {
		buf[i] = byte(x) | 0x80
		x >>= 7
		i++
	}
	buf[i] = byte(x)
	return i + 1
}
