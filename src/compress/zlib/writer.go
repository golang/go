// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zlib

import (
	"compress/flate"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/adler32"
	"io"
)

// These constants are copied from the flate package, so that code that imports
// "compress/zlib" does not also have to import "compress/flate".
const (
	NoCompression      = flate.NoCompression
	BestSpeed          = flate.BestSpeed
	BestCompression    = flate.BestCompression
	DefaultCompression = flate.DefaultCompression
	HuffmanOnly        = flate.HuffmanOnly
)

// A Writer takes data written to it and writes the compressed
// form of that data to an underlying writer (see NewWriter).
type Writer struct {
	w           io.Writer
	level       int
	dict        []byte
	compressor  *flate.Writer
	digest      hash.Hash32
	err         error
	scratch     [4]byte
	wroteHeader bool
}

// NewWriter creates a new Writer.
// Writes to the returned Writer are compressed and written to w.
//
// It is the caller's responsibility to call Close on the Writer when done.
// Writes may be buffered and not flushed until Close.
func NewWriter(w io.Writer) *Writer {
	z, _ := NewWriterLevelDict(w, DefaultCompression, nil)
	return z
}

// NewWriterLevel is like NewWriter but specifies the compression level instead
// of assuming DefaultCompression.
//
// The compression level can be DefaultCompression, NoCompression, HuffmanOnly
// or any integer value between BestSpeed and BestCompression inclusive.
// The error returned will be nil if the level is valid.
func NewWriterLevel(w io.Writer, level int) (*Writer, error) {
	return NewWriterLevelDict(w, level, nil)
}

// NewWriterLevelDict is like NewWriterLevel but specifies a dictionary to
// compress with.
//
// The dictionary may be nil. If not, its contents should not be modified until
// the Writer is closed.
func NewWriterLevelDict(w io.Writer, level int, dict []byte) (*Writer, error) {
	if level < HuffmanOnly || level > BestCompression {
		return nil, fmt.Errorf("zlib: invalid compression level: %d", level)
	}
	return &Writer{
		w:     w,
		level: level,
		dict:  dict,
	}, nil
}

// Reset clears the state of the Writer z such that it is equivalent to its
// initial state from NewWriterLevel or NewWriterLevelDict, but instead writing
// to w.
func (z *Writer) Reset(w io.Writer) {
	z.w = w
	// z.level and z.dict left unchanged.
	if z.compressor != nil {
		z.compressor.Reset(w)
	}
	if z.digest != nil {
		z.digest.Reset()
	}
	z.err = nil
	z.scratch = [4]byte{}
	z.wroteHeader = false
}

// writeHeader writes the ZLIB header.
func (z *Writer) writeHeader() (err error) {
	z.wroteHeader = true
	// ZLIB has a two-byte header (as documented in RFC 1950).
	// The first four bits is the CINFO (compression info), which is 7 for the default deflate window size.
	// The next four bits is the CM (compression method), which is 8 for deflate.
	z.scratch[0] = 0x78
	// The next two bits is the FLEVEL (compression level). The four values are:
	// 0=fastest, 1=fast, 2=default, 3=best.
	// The next bit, FDICT, is set if a dictionary is given.
	// The final five FCHECK bits form a mod-31 checksum.
	switch z.level {
	case -2, 0, 1:
		z.scratch[1] = 0 << 6
	case 2, 3, 4, 5:
		z.scratch[1] = 1 << 6
	case 6, -1:
		z.scratch[1] = 2 << 6
	case 7, 8, 9:
		z.scratch[1] = 3 << 6
	default:
		panic("unreachable")
	}
	if z.dict != nil {
		z.scratch[1] |= 1 << 5
	}
	z.scratch[1] += uint8(31 - (uint16(z.scratch[0])<<8+uint16(z.scratch[1]))%31)
	if _, err = z.w.Write(z.scratch[0:2]); err != nil {
		return err
	}
	if z.dict != nil {
		// The next four bytes are the Adler-32 checksum of the dictionary.
		binary.BigEndian.PutUint32(z.scratch[:], adler32.Checksum(z.dict))
		if _, err = z.w.Write(z.scratch[0:4]); err != nil {
			return err
		}
	}
	if z.compressor == nil {
		// Initialize deflater unless the Writer is being reused
		// after a Reset call.
		z.compressor, err = flate.NewWriterDict(z.w, z.level, z.dict)
		if err != nil {
			return err
		}
		z.digest = adler32.New()
	}
	return nil
}

// Write writes a compressed form of p to the underlying io.Writer. The
// compressed bytes are not necessarily flushed until the Writer is closed or
// explicitly flushed.
func (z *Writer) Write(p []byte) (n int, err error) {
	if !z.wroteHeader {
		z.err = z.writeHeader()
	}
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

// Flush flushes the Writer to its underlying io.Writer.
func (z *Writer) Flush() error {
	if !z.wroteHeader {
		z.err = z.writeHeader()
	}
	if z.err != nil {
		return z.err
	}
	z.err = z.compressor.Flush()
	return z.err
}

// Close closes the Writer, flushing any unwritten data to the underlying
// io.Writer, but does not close the underlying io.Writer.
func (z *Writer) Close() error {
	if !z.wroteHeader {
		z.err = z.writeHeader()
	}
	if z.err != nil {
		return z.err
	}
	z.err = z.compressor.Close()
	if z.err != nil {
		return z.err
	}
	checksum := z.digest.Sum32()
	// ZLIB (RFC 1950) is big-endian, unlike GZIP (RFC 1952).
	binary.BigEndian.PutUint32(z.scratch[:], checksum)
	_, z.err = z.w.Write(z.scratch[0:4])
	return z.err
}
