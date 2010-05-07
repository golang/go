// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The zlib package implements reading and writing of zlib
format compressed data, as specified in RFC 1950.

The implementation provides filters that uncompress during reading
and compress during writing.  For example, to write compressed data
to a buffer:

	var b bytes.Buffer
	w, err := zlib.NewWriter(&b)
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
	"hash"
	"hash/adler32"
	"io"
	"os"
)

const zlibDeflate = 8

var ChecksumError os.Error = os.ErrorString("zlib checksum error")
var HeaderError os.Error = os.ErrorString("invalid zlib header")
var UnsupportedError os.Error = os.ErrorString("unsupported zlib format")

type reader struct {
	r            flate.Reader
	decompressor io.ReadCloser
	digest       hash.Hash32
	err          os.Error
	scratch      [4]byte
}

// NewReader creates a new io.ReadCloser that satisfies reads by decompressing data read from r.
// The implementation buffers input and may read more data than necessary from r.
// It is the caller's responsibility to call Close on the ReadCloser when done.
func NewReader(r io.Reader) (io.ReadCloser, os.Error) {
	z := new(reader)
	if fr, ok := r.(flate.Reader); ok {
		z.r = fr
	} else {
		z.r = bufio.NewReader(r)
	}
	_, err := io.ReadFull(z.r, z.scratch[0:2])
	if err != nil {
		return nil, err
	}
	h := uint(z.scratch[0])<<8 | uint(z.scratch[1])
	if (z.scratch[0]&0x0f != zlibDeflate) || (h%31 != 0) {
		return nil, HeaderError
	}
	if z.scratch[1]&0x20 != 0 {
		// BUG(nigeltao): The zlib package does not implement the FDICT flag.
		return nil, UnsupportedError
	}
	z.digest = adler32.New()
	z.decompressor = flate.NewReader(z.r)
	return z, nil
}

func (z *reader) Read(p []byte) (n int, err os.Error) {
	if z.err != nil {
		return 0, z.err
	}
	if len(p) == 0 {
		return 0, nil
	}

	n, err = z.decompressor.Read(p)
	z.digest.Write(p[0:n])
	if n != 0 || err != os.EOF {
		z.err = err
		return
	}

	// Finished file; check checksum.
	if _, err := io.ReadFull(z.r, z.scratch[0:4]); err != nil {
		z.err = err
		return 0, err
	}
	// ZLIB (RFC 1950) is big-endian, unlike GZIP (RFC 1952).
	checksum := uint32(z.scratch[0])<<24 | uint32(z.scratch[1])<<16 | uint32(z.scratch[2])<<8 | uint32(z.scratch[3])
	if checksum != z.digest.Sum32() {
		z.err = ChecksumError
		return 0, z.err
	}
	return
}

// Calling Close does not close the wrapped io.Reader originally passed to NewReader.
func (z *reader) Close() os.Error {
	if z.err != nil {
		return z.err
	}
	z.err = z.decompressor.Close()
	return z.err
}
