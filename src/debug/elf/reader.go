// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elf

import (
	"io"
	"os"
)

// errorReader returns error from all operations.
type errorReader struct {
	error
}

func (r errorReader) Read(p []byte) (n int, err error) {
	return 0, r.error
}

func (r errorReader) ReadAt(p []byte, off int64) (n int, err error) {
	return 0, r.error
}

func (r errorReader) Seek(offset int64, whence int) (int64, error) {
	return 0, r.error
}

func (r errorReader) Close() error {
	return r.error
}

// readSeekerFromReader converts an io.Reader into an io.ReadSeeker.
// In general Seek may not be efficient, but it is optimized for
// common cases such as seeking to the end to find the length of the
// data.
type readSeekerFromReader struct {
	reset  func() (io.Reader, error)
	r      io.Reader
	size   int64
	offset int64
}

func (r *readSeekerFromReader) start() {
	x, err := r.reset()
	if err != nil {
		r.r = errorReader{err}
	} else {
		r.r = x
	}
	r.offset = 0
}

func (r *readSeekerFromReader) Read(p []byte) (n int, err error) {
	if r.r == nil {
		r.start()
	}
	n, err = r.r.Read(p)
	r.offset += int64(n)
	return n, err
}

func (r *readSeekerFromReader) Seek(offset int64, whence int) (int64, error) {
	var newOffset int64
	switch whence {
	case 0:
		newOffset = offset
	case 1:
		newOffset = r.offset + offset
	case 2:
		newOffset = r.size + offset
	default:
		return 0, os.ErrInvalid
	}

	switch {
	case newOffset == r.offset:
		return newOffset, nil

	case newOffset < 0, newOffset > r.size:
		return 0, os.ErrInvalid

	case newOffset == 0:
		r.r = nil

	case newOffset == r.size:
		r.r = errorReader{io.EOF}

	default:
		if newOffset < r.offset {
			// Restart at the beginning.
			r.start()
		}
		// Read until we reach offset.
		var buf [512]byte
		for r.offset < newOffset {
			b := buf[:]
			if newOffset-r.offset < int64(len(buf)) {
				b = buf[:newOffset-r.offset]
			}
			if _, err := r.Read(b); err != nil {
				return 0, err
			}
		}
	}
	r.offset = newOffset
	return r.offset, nil
}
