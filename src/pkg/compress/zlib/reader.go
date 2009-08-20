// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The zlib package implements reading (and eventually writing) of
// zlib format compressed files, as specified in RFC 1950.
package zlib

import (
	"bufio";
	"compress/flate";
	"hash";
	"hash/adler32";
	"io";
	"os";
)

const zlibDeflate = 8

var ChecksumError os.Error = os.ErrorString("zlib checksum error")
var HeaderError os.Error = os.ErrorString("invalid zlib header")
var UnsupportedError os.Error = os.ErrorString("unsupported zlib format")

type reader struct {
	r flate.Reader;
	inflater io.Reader;
	digest hash.Hash32;
	err os.Error;
}

// NewInflater creates a new io.Reader that satisfies reads by decompressing data read from r.
// The implementation buffers input and may read more data than necessary from r.
func NewInflater(r io.Reader) (io.Reader, os.Error) {
	z := new(reader);
	if fr, ok := r.(flate.Reader); ok {
		z.r = fr;
	} else {
		z.r = bufio.NewReader(r);
	}
	var buf [2]byte;
	n, err := io.ReadFull(z.r, buf[0:2]);
	if err != nil {
		return nil, err;
	}
	h := uint(buf[0])<<8 | uint(buf[1]);
	if (buf[0] & 0x0f != zlibDeflate) || (h % 31 != 0) {
		return nil, HeaderError;
	}
	if buf[1] & 0x20 != 0 {
		// BUG(nigeltao): The zlib package does not implement the FDICT flag.
		return nil, UnsupportedError;
	}
	z.digest = adler32.New();
	z.inflater = flate.NewInflater(z.r);
	return z, nil;
}

func (z *reader) Read(p []byte) (n int, err os.Error) {
	if z.err != nil {
		return 0, z.err;
	}
	if len(p) == 0 {
		return 0, nil;
	}

	n, err = z.inflater.Read(p);
	z.digest.Write(p[0:n]);
	if n != 0 || err != os.EOF {
		z.err = err;
		return;
	}

	// Finished file; check checksum.
	var buf [4]byte;
	if _, err := io.ReadFull(z.r, buf[0:4]); err != nil {
		z.err = err;
		return 0, err;
	}
	// ZLIB (RFC 1950) is big-endian, unlike GZIP (RFC 1952).
	checksum := uint32(buf[0])<<24 | uint32(buf[1])<<16 | uint32(buf[2])<<8 | uint32(buf[3]);
	if checksum != z.digest.Sum32() {
		z.err = ChecksumError;
		return 0, z.err;
	}
	return;
}

