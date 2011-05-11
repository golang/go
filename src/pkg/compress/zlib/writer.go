// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zlib

import (
	"compress/flate"
	"hash"
	"hash/adler32"
	"io"
	"os"
)

// These constants are copied from the flate package, so that code that imports
// "compress/zlib" does not also have to import "compress/flate".
const (
	NoCompression      = flate.NoCompression
	BestSpeed          = flate.BestSpeed
	BestCompression    = flate.BestCompression
	DefaultCompression = flate.DefaultCompression
)

// A Writer takes data written to it and writes the compressed
// form of that data to an underlying writer (see NewWriter).
type Writer struct {
	w          io.Writer
	compressor *flate.Writer
	digest     hash.Hash32
	err        os.Error
	scratch    [4]byte
}

// NewWriter calls NewWriterLevel with the default compression level.
func NewWriter(w io.Writer) (*Writer, os.Error) {
	return NewWriterLevel(w, DefaultCompression)
}

// NewWriterLevel calls NewWriterDict with no dictionary.
func NewWriterLevel(w io.Writer, level int) (*Writer, os.Error) {
	return NewWriterDict(w, level, nil)
}

// NewWriterDict creates a new io.WriteCloser that satisfies writes by compressing data written to w.
// It is the caller's responsibility to call Close on the WriteCloser when done.
// level is the compression level, which can be DefaultCompression, NoCompression,
// or any integer value between BestSpeed and BestCompression (inclusive).
// dict is the preset dictionary to compress with, or nil to use no dictionary.
func NewWriterDict(w io.Writer, level int, dict []byte) (*Writer, os.Error) {
	z := new(Writer)
	// ZLIB has a two-byte header (as documented in RFC 1950).
	// The first four bits is the CINFO (compression info), which is 7 for the default deflate window size.
	// The next four bits is the CM (compression method), which is 8 for deflate.
	z.scratch[0] = 0x78
	// The next two bits is the FLEVEL (compression level). The four values are:
	// 0=fastest, 1=fast, 2=default, 3=best.
	// The next bit, FDICT, is set if a dictionary is given.
	// The final five FCHECK bits form a mod-31 checksum.
	switch level {
	case 0, 1:
		z.scratch[1] = 0 << 6
	case 2, 3, 4, 5:
		z.scratch[1] = 1 << 6
	case 6, -1:
		z.scratch[1] = 2 << 6
	case 7, 8, 9:
		z.scratch[1] = 3 << 6
	default:
		return nil, os.NewError("level out of range")
	}
	if dict != nil {
		z.scratch[1] |= 1 << 5
	}
	z.scratch[1] += uint8(31 - (uint16(z.scratch[0])<<8+uint16(z.scratch[1]))%31)
	_, err := w.Write(z.scratch[0:2])
	if err != nil {
		return nil, err
	}
	if dict != nil {
		// The next four bytes are the Adler-32 checksum of the dictionary.
		checksum := adler32.Checksum(dict)
		z.scratch[0] = uint8(checksum >> 24)
		z.scratch[1] = uint8(checksum >> 16)
		z.scratch[2] = uint8(checksum >> 8)
		z.scratch[3] = uint8(checksum >> 0)
		_, err = w.Write(z.scratch[0:4])
		if err != nil {
			return nil, err
		}
	}
	z.w = w
	z.compressor = flate.NewWriterDict(w, level, dict)
	z.digest = adler32.New()
	return z, nil
}

func (z *Writer) Write(p []byte) (n int, err os.Error) {
	if z.err != nil {
		return 0, z.err
	}
	if len(p) == 0 {
		return 0, nil
	}
	n, err = z.compressor.Write(p)
	if err != nil {
		z.err = err
		return
	}
	z.digest.Write(p)
	return
}

// Flush flushes the underlying compressor.
func (z *Writer) Flush() os.Error {
	if z.err != nil {
		return z.err
	}
	z.err = z.compressor.Flush()
	return z.err
}

// Calling Close does not close the wrapped io.Writer originally passed to NewWriter.
func (z *Writer) Close() os.Error {
	if z.err != nil {
		return z.err
	}
	z.err = z.compressor.Close()
	if z.err != nil {
		return z.err
	}
	checksum := z.digest.Sum32()
	// ZLIB (RFC 1950) is big-endian, unlike GZIP (RFC 1952).
	z.scratch[0] = uint8(checksum >> 24)
	z.scratch[1] = uint8(checksum >> 16)
	z.scratch[2] = uint8(checksum >> 8)
	z.scratch[3] = uint8(checksum >> 0)
	_, z.err = z.w.Write(z.scratch[0:4])
	return z.err
}
