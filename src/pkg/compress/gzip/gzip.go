// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gzip

import (
	"compress/flate"
	"errors"
	"hash"
	"hash/crc32"
	"io"
)

// These constants are copied from the flate package, so that code that imports
// "compress/gzip" does not also have to import "compress/flate".
const (
	NoCompression      = flate.NoCompression
	BestSpeed          = flate.BestSpeed
	BestCompression    = flate.BestCompression
	DefaultCompression = flate.DefaultCompression
)

// A Compressor is an io.WriteCloser that satisfies writes by compressing data written
// to its wrapped io.Writer.
type Compressor struct {
	Header
	w          io.Writer
	level      int
	compressor io.WriteCloser
	digest     hash.Hash32
	size       uint32
	closed     bool
	buf        [10]byte
	err        error
}

// NewWriter calls NewWriterLevel with the default compression level.
func NewWriter(w io.Writer) (*Compressor, error) {
	return NewWriterLevel(w, DefaultCompression)
}

// NewWriterLevel creates a new Compressor writing to the given writer.
// Writes may be buffered and not flushed until Close.
// Callers that wish to set the fields in Compressor.Header must
// do so before the first call to Write or Close.
// It is the caller's responsibility to call Close on the WriteCloser when done.
// level is the compression level, which can be DefaultCompression, NoCompression,
// or any integer value between BestSpeed and BestCompression (inclusive).
func NewWriterLevel(w io.Writer, level int) (*Compressor, error) {
	z := new(Compressor)
	z.OS = 255 // unknown
	z.w = w
	z.level = level
	z.digest = crc32.NewIEEE()
	return z, nil
}

// GZIP (RFC 1952) is little-endian, unlike ZLIB (RFC 1950).
func put2(p []byte, v uint16) {
	p[0] = uint8(v >> 0)
	p[1] = uint8(v >> 8)
}

func put4(p []byte, v uint32) {
	p[0] = uint8(v >> 0)
	p[1] = uint8(v >> 8)
	p[2] = uint8(v >> 16)
	p[3] = uint8(v >> 24)
}

// writeBytes writes a length-prefixed byte slice to z.w.
func (z *Compressor) writeBytes(b []byte) error {
	if len(b) > 0xffff {
		return errors.New("gzip.Write: Extra data is too large")
	}
	put2(z.buf[0:2], uint16(len(b)))
	_, err := z.w.Write(z.buf[0:2])
	if err != nil {
		return err
	}
	_, err = z.w.Write(b)
	return err
}

// writeString writes a string (in ISO 8859-1 (Latin-1) format) to z.w.
func (z *Compressor) writeString(s string) error {
	// GZIP (RFC 1952) specifies that strings are NUL-terminated ISO 8859-1 (Latin-1).
	var err error
	needconv := false
	for _, v := range s {
		if v == 0 || v > 0xff {
			return errors.New("gzip.Write: non-Latin-1 header string")
		}
		if v > 0x7f {
			needconv = true
		}
	}
	if needconv {
		b := make([]byte, 0, len(s))
		for _, v := range s {
			b = append(b, byte(v))
		}
		_, err = z.w.Write(b)
	} else {
		_, err = io.WriteString(z.w, s)
	}
	if err != nil {
		return err
	}
	// GZIP strings are NUL-terminated.
	z.buf[0] = 0
	_, err = z.w.Write(z.buf[0:1])
	return err
}

func (z *Compressor) Write(p []byte) (int, error) {
	if z.err != nil {
		return 0, z.err
	}
	var n int
	// Write the GZIP header lazily.
	if z.compressor == nil {
		z.buf[0] = gzipID1
		z.buf[1] = gzipID2
		z.buf[2] = gzipDeflate
		z.buf[3] = 0
		if z.Extra != nil {
			z.buf[3] |= 0x04
		}
		if z.Name != "" {
			z.buf[3] |= 0x08
		}
		if z.Comment != "" {
			z.buf[3] |= 0x10
		}
		put4(z.buf[4:8], uint32(z.ModTime.Unix()))
		if z.level == BestCompression {
			z.buf[8] = 2
		} else if z.level == BestSpeed {
			z.buf[8] = 4
		} else {
			z.buf[8] = 0
		}
		z.buf[9] = z.OS
		n, z.err = z.w.Write(z.buf[0:10])
		if z.err != nil {
			return n, z.err
		}
		if z.Extra != nil {
			z.err = z.writeBytes(z.Extra)
			if z.err != nil {
				return n, z.err
			}
		}
		if z.Name != "" {
			z.err = z.writeString(z.Name)
			if z.err != nil {
				return n, z.err
			}
		}
		if z.Comment != "" {
			z.err = z.writeString(z.Comment)
			if z.err != nil {
				return n, z.err
			}
		}
		z.compressor = flate.NewWriter(z.w, z.level)
	}
	z.size += uint32(len(p))
	z.digest.Write(p)
	n, z.err = z.compressor.Write(p)
	return n, z.err
}

// Calling Close does not close the wrapped io.Writer originally passed to NewWriter.
func (z *Compressor) Close() error {
	if z.err != nil {
		return z.err
	}
	if z.closed {
		return nil
	}
	z.closed = true
	if z.compressor == nil {
		z.Write(nil)
		if z.err != nil {
			return z.err
		}
	}
	z.err = z.compressor.Close()
	if z.err != nil {
		return z.err
	}
	put4(z.buf[0:4], z.digest.Sum32())
	put4(z.buf[4:8], z.size)
	_, z.err = z.w.Write(z.buf[0:8])
	return z.err
}
