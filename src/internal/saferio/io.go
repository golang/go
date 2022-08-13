// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package saferio provides I/O functions that avoid allocating large
// amounts of memory unnecessarily. This is intended for packages that
// read data from an [io.Reader] where the size is part of the input
// data but the input may be corrupt, or may be provided by an
// untrustworthy attacker.
package saferio

import "io"

// chunk is an arbitrary limit on how much memory we are willing
// to allocate without concern.
const chunk = 10 << 20 // 10M

// ReadData reads n bytes from the input stream, but avoids allocating
// all n bytes if n is large. This avoids crashing the program by
// allocating all n bytes in cases where n is incorrect.
func ReadData(r io.Reader, n uint64) ([]byte, error) {
	if int64(n) < 0 || n != uint64(int(n)) {
		// n is too large to fit in int, so we can't allocate
		// a buffer large enough. Treat this as a read failure.
		return nil, io.ErrUnexpectedEOF
	}

	if n < chunk {
		buf := make([]byte, n)
		_, err := io.ReadFull(r, buf)
		if err != nil {
			return nil, err
		}
		return buf, nil
	}

	var buf []byte
	buf1 := make([]byte, chunk)
	for n > 0 {
		next := n
		if next > chunk {
			next = chunk
		}
		_, err := io.ReadFull(r, buf1[:next])
		if err != nil {
			return nil, err
		}
		buf = append(buf, buf1[:next]...)
		n -= next
	}
	return buf, nil
}
