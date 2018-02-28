// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package zlib implements reading and writing of zlib format compressed data,
as specified in RFC 1950.

The implementation provides filters that uncompress during reading
and compress during writing.  For example, to write compressed data
to a buffer:

	var b bytes.Buffer
	w := zlib.NewWriter(&b)
	w.Write([]byte("hello, world\n"))
	w.Close()

and to read that data back:

	r, err := zlib.NewReader(&b)
	io.Copy(os.Stdout, r)
	r.Close()
*/
package zlib

import (
	"bufio"
	"compress/flate"
	"errors"
	"hash"
	"hash/adler32"
	"io"
)

const zlibDeflate = 8

var (
	// ErrChecksum is returned when reading ZLIB data that has an invalid checksum.
	ErrChecksum = errors.New("zlib: invalid checksum")
	// ErrDictionary is returned when reading ZLIB data that has an invalid dictionary.
	ErrDictionary = errors.New("zlib: invalid dictionary")
	// ErrHeader is returned when reading ZLIB data that has an invalid header.
	ErrHeader = errors.New("zlib: invalid header")
)

type reader struct {
	r            flate.Reader
	decompressor io.ReadCloser
	digest       hash.Hash32
	err          error
	scratch      [4]byte
}

// Resetter resets a ReadCloser returned by NewReader or NewReaderDict to
// to switch to a new underlying Reader. This permits reusing a ReadCloser
// instead of allocating a new one.
type Resetter interface {
	// Reset discards any buffered data and resets the Resetter as if it was
	// newly initialized with the given reader.
	Reset(r io.Reader, dict []byte) error
}

// NewReader creates a new ReadCloser.
// Reads from the returned ReadCloser read and decompress data from r.
// If r does not implement io.ByteReader, the decompressor may read more
// data than necessary from r.
// It is the caller's responsibility to call Close on the ReadCloser when done.
//
// The ReadCloser returned by NewReader also implements Resetter.
func NewReader(r io.Reader) (io.ReadCloser, error) {
	return NewReaderDict(r, nil)
}

// NewReaderDict is like NewReader but uses a preset dictionary.
// NewReaderDict ignores the dictionary if the compressed data does not refer to it.
// If the compressed data refers to a different dictionary, NewReaderDict returns ErrDictionary.
//
// The ReadCloser returned by NewReaderDict also implements Resetter.
func NewReaderDict(r io.Reader, dict []byte) (io.ReadCloser, error) {
	z := new(reader)
	err := z.Reset(r, dict)
	if err != nil {
		return nil, err
	}
	return z, nil
}

func (z *reader) Read(p []byte) (int, error) {
	if z.err != nil {
		return 0, z.err
	}

	var n int
	n, z.err = z.decompressor.Read(p)
	z.digest.Write(p[0:n])
	if z.err != io.EOF {
		// In the normal case we return here.
		return n, z.err
	}

	// Finished file; check checksum.
	if _, err := io.ReadFull(z.r, z.scratch[0:4]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		z.err = err
		return n, z.err
	}
	// ZLIB (RFC 1950) is big-endian, unlike GZIP (RFC 1952).
	checksum := uint32(z.scratch[0])<<24 | uint32(z.scratch[1])<<16 | uint32(z.scratch[2])<<8 | uint32(z.scratch[3])
	if checksum != z.digest.Sum32() {
		z.err = ErrChecksum
		return n, z.err
	}
	return n, io.EOF
}

// Calling Close does not close the wrapped io.Reader originally passed to NewReader.
// In order for the ZLIB checksum to be verified, the reader must be
// fully consumed until the io.EOF.
func (z *reader) Close() error {
	if z.err != nil && z.err != io.EOF {
		return z.err
	}
	z.err = z.decompressor.Close()
	return z.err
}

func (z *reader) Reset(r io.Reader, dict []byte) error {
	*z = reader{decompressor: z.decompressor}
	if fr, ok := r.(flate.Reader); ok {
		z.r = fr
	} else {
		z.r = bufio.NewReader(r)
	}

	// Read the header (RFC 1950 section 2.2.).
	_, z.err = io.ReadFull(z.r, z.scratch[0:2])
	if z.err != nil {
		if z.err == io.EOF {
			z.err = io.ErrUnexpectedEOF
		}
		return z.err
	}
	h := uint(z.scratch[0])<<8 | uint(z.scratch[1])
	if (z.scratch[0]&0x0f != zlibDeflate) || (h%31 != 0) {
		z.err = ErrHeader
		return z.err
	}
	haveDict := z.scratch[1]&0x20 != 0
	if haveDict {
		_, z.err = io.ReadFull(z.r, z.scratch[0:4])
		if z.err != nil {
			if z.err == io.EOF {
				z.err = io.ErrUnexpectedEOF
			}
			return z.err
		}
		checksum := uint32(z.scratch[0])<<24 | uint32(z.scratch[1])<<16 | uint32(z.scratch[2])<<8 | uint32(z.scratch[3])
		if checksum != adler32.Checksum(dict) {
			z.err = ErrDictionary
			return z.err
		}
	}

	if z.decompressor == nil {
		if haveDict {
			z.decompressor = flate.NewReaderDict(z.r, dict)
		} else {
			z.decompressor = flate.NewReader(z.r)
		}
	} else {
		z.decompressor.(flate.Resetter).Reset(z.r, dict)
	}
	z.digest = adler32.New()
	return nil
}
