// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gzip implements reading and writing of gzip format compressed files,
// as specified in RFC 1952.
package gzip

import (
	"bufio"
	"compress/flate"
	"errors"
	"hash"
	"hash/crc32"
	"io"
	"time"
)

const (
	gzipID1     = 0x1f
	gzipID2     = 0x8b
	gzipDeflate = 8
	flagText    = 1 << 0
	flagHdrCrc  = 1 << 1
	flagExtra   = 1 << 2
	flagName    = 1 << 3
	flagComment = 1 << 4
)

func makeReader(r io.Reader) flate.Reader {
	if rr, ok := r.(flate.Reader); ok {
		return rr
	}
	return bufio.NewReader(r)
}

var (
	// ErrChecksum is returned when reading GZIP data that has an invalid checksum.
	ErrChecksum = errors.New("gzip: invalid checksum")
	// ErrHeader is returned when reading GZIP data that has an invalid header.
	ErrHeader = errors.New("gzip: invalid header")
)

// The gzip file stores a header giving metadata about the compressed file.
// That header is exposed as the fields of the Writer and Reader structs.
type Header struct {
	Comment string    // comment
	Extra   []byte    // "extra data"
	ModTime time.Time // modification time
	Name    string    // file name
	OS      byte      // operating system type
}

// A Reader is an io.Reader that can be read to retrieve
// uncompressed data from a gzip-format compressed file.
//
// In general, a gzip file can be a concatenation of gzip files,
// each with its own header.  Reads from the Reader
// return the concatenation of the uncompressed data of each.
// Only the first header is recorded in the Reader fields.
//
// Gzip files store a length and checksum of the uncompressed data.
// The Reader will return a ErrChecksum when Read
// reaches the end of the uncompressed data if it does not
// have the expected length or checksum.  Clients should treat data
// returned by Read as tentative until they receive the io.EOF
// marking the end of the data.
type Reader struct {
	Header
	r            flate.Reader
	decompressor io.ReadCloser
	digest       hash.Hash32
	size         uint32
	flg          byte
	buf          [512]byte
	err          error
	multistream  bool
}

// NewReader creates a new Reader reading the given reader.
// If r does not also implement io.ByteReader,
// the decompressor may read more data than necessary from r.
// It is the caller's responsibility to call Close on the Reader when done.
func NewReader(r io.Reader) (*Reader, error) {
	z := new(Reader)
	z.r = makeReader(r)
	z.multistream = true
	z.digest = crc32.NewIEEE()
	if err := z.readHeader(true); err != nil {
		return nil, err
	}
	return z, nil
}

// Reset discards the Reader z's state and makes it equivalent to the
// result of its original state from NewReader, but reading from r instead.
// This permits reusing a Reader rather than allocating a new one.
func (z *Reader) Reset(r io.Reader) error {
	z.r = makeReader(r)
	if z.digest == nil {
		z.digest = crc32.NewIEEE()
	} else {
		z.digest.Reset()
	}
	z.size = 0
	z.err = nil
	z.multistream = true
	return z.readHeader(true)
}

// Multistream controls whether the reader supports multistream files.
//
// If enabled (the default), the Reader expects the input to be a sequence
// of individually gzipped data streams, each with its own header and
// trailer, ending at EOF. The effect is that the concatenation of a sequence
// of gzipped files is treated as equivalent to the gzip of the concatenation
// of the sequence. This is standard behavior for gzip readers.
//
// Calling Multistream(false) disables this behavior; disabling the behavior
// can be useful when reading file formats that distinguish individual gzip
// data streams or mix gzip data streams with other data streams.
// In this mode, when the Reader reaches the end of the data stream,
// Read returns io.EOF. If the underlying reader implements io.ByteReader,
// it will be left positioned just after the gzip stream.
// To start the next stream, call z.Reset(r) followed by z.Multistream(false).
// If there is no next stream, z.Reset(r) will return io.EOF.
func (z *Reader) Multistream(ok bool) {
	z.multistream = ok
}

// GZIP (RFC 1952) is little-endian, unlike ZLIB (RFC 1950).
func get4(p []byte) uint32 {
	return uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16 | uint32(p[3])<<24
}

func (z *Reader) readString() (string, error) {
	var err error
	needconv := false
	for i := 0; ; i++ {
		if i >= len(z.buf) {
			return "", ErrHeader
		}
		z.buf[i], err = z.r.ReadByte()
		if err != nil {
			return "", err
		}
		if z.buf[i] > 0x7f {
			needconv = true
		}
		if z.buf[i] == 0 {
			// GZIP (RFC 1952) specifies that strings are NUL-terminated ISO 8859-1 (Latin-1).
			if needconv {
				s := make([]rune, 0, i)
				for _, v := range z.buf[0:i] {
					s = append(s, rune(v))
				}
				return string(s), nil
			}
			return string(z.buf[0:i]), nil
		}
	}
}

func (z *Reader) read2() (uint32, error) {
	_, err := io.ReadFull(z.r, z.buf[0:2])
	if err != nil {
		return 0, err
	}
	return uint32(z.buf[0]) | uint32(z.buf[1])<<8, nil
}

func (z *Reader) readHeader(save bool) error {
	_, err := io.ReadFull(z.r, z.buf[0:10])
	if err != nil {
		return err
	}
	if z.buf[0] != gzipID1 || z.buf[1] != gzipID2 || z.buf[2] != gzipDeflate {
		return ErrHeader
	}
	z.flg = z.buf[3]
	if save {
		z.ModTime = time.Unix(int64(get4(z.buf[4:8])), 0)
		// z.buf[8] is xfl, ignored
		z.OS = z.buf[9]
	}
	z.digest.Reset()
	z.digest.Write(z.buf[0:10])

	if z.flg&flagExtra != 0 {
		n, err := z.read2()
		if err != nil {
			return err
		}
		data := make([]byte, n)
		if _, err = io.ReadFull(z.r, data); err != nil {
			return err
		}
		if save {
			z.Extra = data
		}
	}

	var s string
	if z.flg&flagName != 0 {
		if s, err = z.readString(); err != nil {
			return err
		}
		if save {
			z.Name = s
		}
	}

	if z.flg&flagComment != 0 {
		if s, err = z.readString(); err != nil {
			return err
		}
		if save {
			z.Comment = s
		}
	}

	if z.flg&flagHdrCrc != 0 {
		n, err := z.read2()
		if err != nil {
			return err
		}
		sum := z.digest.Sum32() & 0xFFFF
		if n != sum {
			return ErrHeader
		}
	}

	z.digest.Reset()
	if z.decompressor == nil {
		z.decompressor = flate.NewReader(z.r)
	} else {
		z.decompressor.(flate.Resetter).Reset(z.r, nil)
	}
	return nil
}

func (z *Reader) Read(p []byte) (n int, err error) {
	if z.err != nil {
		return 0, z.err
	}
	if len(p) == 0 {
		return 0, nil
	}

	n, err = z.decompressor.Read(p)
	z.digest.Write(p[0:n])
	z.size += uint32(n)
	if n != 0 || err != io.EOF {
		z.err = err
		return
	}

	// Finished file; check checksum + size.
	if _, err := io.ReadFull(z.r, z.buf[0:8]); err != nil {
		z.err = err
		return 0, err
	}
	crc32, isize := get4(z.buf[0:4]), get4(z.buf[4:8])
	sum := z.digest.Sum32()
	if sum != crc32 || isize != z.size {
		z.err = ErrChecksum
		return 0, z.err
	}

	// File is ok; is there another?
	if !z.multistream {
		return 0, io.EOF
	}

	if err = z.readHeader(false); err != nil {
		z.err = err
		return
	}

	// Yes.  Reset and read from it.
	z.digest.Reset()
	z.size = 0
	return z.Read(p)
}

// Close closes the Reader. It does not close the underlying io.Reader.
func (z *Reader) Close() error { return z.decompressor.Close() }
